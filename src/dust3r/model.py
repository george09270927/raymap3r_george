'''
Based on the model_ori, target on state feat, extend with static ray map input branch, identify dynamic regions with geometry information.

Results: works better than baseline model_ori on depth, worse on pose.

'''


import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.utils.camera import PoseEncoder, pose_encoding_to_camera
from dust3r.post_process import estimate_focal_knowing_depth
import numpy as np
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa
from dust3r.blocks import (
    Block,
    DecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
)  # noqa

inf = float("inf")
from accelerate.logging import get_logger

from einops import rearrange
from dust3r.utils.device import to_cpu, to_gpu

printer = get_logger(__name__, log_level="DEBUG")


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """

    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None


def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR only for aspect ratio not consistent
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class ARCroco3DStereoConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",  # or dpt
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        ray_enc_depth=2,
        state_size=324,
        local_mem_size=256,
        state_pe="2d",
        state_dec_num_heads=16,
        depth_head=False,
        rgb_head=False,
        pose_conf_head=False,
        pose_head=False,
        model_update_type="cut3r",
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        self.state_size = state_size
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.model_update_type = model_update_type
        self.croco_kwargs = croco_kwargs


class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        ) # [1, 1, 768] pose mask token
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        ) # [1, 256, 1536] pose mem
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v, return_attn=False):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C] global_img_feat
        feat_v: [B, 1, C] out_pose_feat
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)

        attention_maps = []
        for blk in self.write_blocks:
            mem, _, self_attn, cross_attn = blk(mem, feat, None, None, return_attn=return_attn)
            attention_maps.append((self_attn, cross_attn))
        return mem

    def inquire(self, query, mem, return_attn=False):
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1) # [1, 1, 768 global_img_feat_i + 768 masked_token(pose)]
        attention_maps = []
        for blk in self.read_blocks:
            x, _, self_attn, cross_attn = blk(x, mem, None, None, return_attn=return_attn)
            attention_maps.append((self_attn, cross_attn))
        return x[..., -self.v_dim :]


class ARCroco3DStereo(CroCoNet):
    config_class = ARCroco3DStereoConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True

    def __init__(self, config: ARCroco3DStereoConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=self.rope,
                )
                for _ in range(config.ray_enc_depth)
            ]
        )
        self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            ) # [1, 1, 768]
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            )
        self.register_tokens = nn.Embedding(config.state_size, self.enc_embed_dim) # init state tokens [768, 1024]
        self.state_size = config.state_size
        self.state_pe = config.state_pe
        self.masked_img_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.masked_ray_map_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.state_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )
        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.rgb_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        self.set_freeze(config.freeze)

        # gating hyperparameters (decoupled strength and floor)
        self.alpha_gate_lambda = 0.9  # strength of alpha gating 0.5
        self.alpha_gate_wmin = 0.15   # gate floor to avoid freezing tokens
        self.alpha_ema_tau = 5.0      # EMA window for temporal smoothing
        self.coverage_adapt_k = 0.5   # reduce lambda when dynamic coverage is high
        self.small_step_c = 0.5       # reduce step size when dynamic coverage is high
        self._alpha_ema = None        # temporal buffer
        # static-key mixing ratio (for mem update): blend mean key with static-weighted key
        self.static_key_mix_beta = 0.3

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(ARCroco3DStereo, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_state") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_state")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token] if hasattr(self, "mask_token") else [],
            "encoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
            ],
            "encoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.downstream_head,
            ],
            "encoder_and_decoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
            ],
            "decoder": [
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_rgb=bool(rgb_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_ray_map(self, ray_map, true_shape):
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks_ray_map:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm_ray_map(x)
        return [x], pos, None

    def _encode_state(self, image_tokens, image_pos):
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        ) # [768, 1024]
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _encode_views(self, views, img_mask=None, ray_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        given = True
        if img_mask is None and ray_mask is None:
            given = False
        if not given:
            img_mask = torch.stack(
                [view["img_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
            ray_mask = torch.stack(
                [view["ray_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        ray_maps = torch.stack(
            [view["ray_map"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, H, W, C)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        ray_maps = ray_maps.view(
            -1, *ray_maps.shape[2:]
        )  # Shape: (num_views * batch_size, H, W, C)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        ray_masks_flat = ray_mask.view(-1)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
            full_out[i][~img_masks_flat] += self.masked_img_token
        full_pos[img_masks_flat] += img_pos
        ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = self._encode_ray_map(
                selected_ray_maps, selected_shapes_ray
            )
            assert len(ray_out) == len(full_out), f"{len(ray_out)}, {len(full_out)}"
            for i in range(len(ray_out)):
                full_out[i][ray_masks_flat] += ray_out[i]
                full_out[i][~ray_masks_flat] += self.masked_ray_map_token
            full_pos[ray_masks_flat] += (
                ray_pos * (~img_masks_flat[ray_masks_flat][:, None, None]).long()
            )
        else:
            raymaps = torch.zeros(
                1, 6, imgs[0].shape[-2], imgs[0].shape[-1], device=img_out[0].device
            )
            ray_mask_flat = torch.zeros_like(img_masks_flat)
            ray_mask_flat[:1] = True
            ray_out, ray_pos, _ = self._encode_ray_map(raymaps, shapes[ray_mask_flat])
            for i in range(len(ray_out)):
                full_out[i][ray_mask_flat] += ray_out[i] * 0.0
                full_out[i][~ray_mask_flat] += self.masked_ray_map_token * 0.0
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
        )

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose, return_attn):
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        f_img = self.decoder_embed(f_img) # Linear: [1, 576, 1024] -> [1, 576, 768]
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1) # [1, 1 + 576, 768]
            pos_img = torch.cat([pos_pose, pos_img], dim=1) # [1, 1 + 576, 2]
        final_output.append((f_state, f_img))
        attention_maps = []
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                f_state, _, self_attn_state, cross_attn_state = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    return_attn,
                    use_reentrant=not self.fixed_input_length,
                )
                f_img, _, self_attn_img, cross_attn_img = checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    return_attn,
                    use_reentrant=not self.fixed_input_length,
                )
            else:
                f_state, _, self_attn_state, cross_attn_state = blk_state(*final_output[-1][::+1], pos_state, pos_img, return_attn=return_attn)
                f_img, _, self_attn_img, cross_attn_img = blk_img(*final_output[-1][::-1], pos_img, pos_state, return_attn=return_attn)
            final_output.append((f_state, f_img))
            attention_maps.append((self_attn_state, cross_attn_state, self_attn_img, cross_attn_img))
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output), zip(*attention_maps)

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        # [1, 768, 768] [1, 768, 2]
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)
        state_feat = self.decoder_embed_state(state_feat) # Linear: [1, 768, 1024] -> [1, 768, 768]
        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
        return_attn=False,
    ):
        (new_state_feat, dec), (self_attn_state, cross_attn_state, self_attn_img, cross_attn_img) = self._decoder(
            state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos, return_attn
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    @torch.no_grad()
    def _static_weighted_global_key_from_alpha_img(self, feat_i: torch.Tensor, alpha_img: torch.Tensor, hpwp: Optional[Tuple[int,int]] = None) -> torch.Tensor:
        """Build a static-weighted global key from per-pixel alpha map.

        Args:
            feat_i: (B, N_img, C) image-token features (encoder space)
            alpha_img: (B, 1, H, W) pixel-level static weights in [0,1]

        Returns:
            global key as (B, 1, C)
        """
        B, N, C = feat_i.shape
        _, _, H, W = alpha_img.shape

        # stable factorization of N into (Hp, Wp)
        def factor_hw(n: int):
            d = int(n ** 0.5)
            for h in range(d, 0, -1):
                if n % h == 0:
                    return h, n // h
            return d, max(1, n // max(1, d))

        if hpwp is None:
            Hp, Wp = factor_hw(N)
        else:
            Hp, Wp = hpwp
            # mild safeguard to avoid shape mismatch
            if Hp * Wp != N:
                Hp, Wp = factor_hw(N)
        a_patch = F.adaptive_avg_pool2d(alpha_img, (Hp, Wp))  # (B,1,Hp,Wp)
        a_patch = a_patch.view(B, N, 1).clamp(0.0, 1.0)        # (B,N,1)
        s = a_patch.sum(dim=1, keepdim=True)                   # (B,1,1)
        # empty-set safeguard: fall back to uniform per-batch item
        w = a_patch / s.clamp_min(1e-6)
        mask = (s < 1e-6).expand_as(w)
        if mask.any():
            w[mask] = 1.0 / float(max(1, N))
        # weighted global pool over tokens: (B,N,C) * (B,N,1) -> (B,1,C)
        g = (feat_i * w).sum(dim=1, keepdim=True)
        return g
    

    @staticmethod
    def _generate_pseudo_intrinsics(h: int, w: int) -> np.ndarray:
        """Generate simple pinhole intrinsics (viser_utils style).
        Focal = sqrt(h^2 + w^2), principal point at ((w-1)/2, (h-1)/2).
        Returns a 3x3 np.float32 matrix.
        """
        focal = float((h ** 2 + w ** 2) ** 0.5)
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        return np.array(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    @staticmethod
    def _ray_map_from_pose(c2w: np.ndarray, h: int, w: int, intrinsics: np.ndarray) -> np.ndarray:
        """Compute a ray map [ro(3), rd(3)] for a single pose.
        - c2w: 4x4 cam2world matrix (np.ndarray)
        - intrinsics: 3x3 (np.ndarray)
        Returns: (H, W, 6) np.float32
        """
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
        grid = np.stack([i, j, np.ones_like(i)], axis=-1)
        ro = c2w[:3, 3]
        rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
        rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
        rd = rd / (np.linalg.norm(rd, axis=-1, keepdims=True) + 1e-8)
        ro = np.broadcast_to(ro, (h, w, 3))
        return np.concatenate([ro, rd], axis=-1).astype(np.float32)

    # ------- Ray-only auxiliary pipeline (no return, cache-only) -------
    @torch.no_grad()

    @torch.no_grad()
    def _remap_raymap(
        self,
        res: dict,
        view: dict,
        state_feat: torch.Tensor,
        state_pos: torch.Tensor,
        init_state_feat: torch.Tensor,
        mem: torch.Tensor,
        device: torch.device,
        override_c2w: 'np.ndarray | None' = None,
    ):
        """Full ray-only pipeline packed as a single callable.
        Steps: (1) build raymap from predicted pose; (2) encode; (3) rollout with OLD state.
        Returns a dict with intermediate tensors for future use.
        override_c2w: optional (B,4,4) np.ndarray to use instead of res["camera_pose"] .
        """
        out = {}
        try:
            if not isinstance(res, dict) or ("camera_pose" not in res):
                return out
            # use override c2w if provided, else compute from res
            if override_c2w is not None:
                c2w_mats = override_c2w
            else:
                cam_pose_enc = res["camera_pose"]  # (B,7)
                c2w_mats = pose_encoding_to_camera(cam_pose_enc).detach().cpu().numpy()  # (B,4,4)
            B = c2w_mats.shape[0]
            if "pts3d_in_self_view" in res:
                H = int(res["pts3d_in_self_view"].shape[1])
                W = int(res["pts3d_in_self_view"].shape[2])
            # estimate focal from the current main-branch 3D
            Ks = None
            try:
                if "pts3d_in_self_view" in res:
                    pts = res["pts3d_in_self_view"]  # (B,H,W,3)
                    Bp, Hp, Wp, _ = pts.shape
                    if (Bp == B) and (Hp == H) and (Wp == W):
                        pp = torch.tensor([W // 2, H // 2], dtype=pts.dtype, device=pts.device).repeat(B, 1)
                        focals = estimate_focal_knowing_depth(pts, pp, focal_mode="weiszfeld")  # (B,)
                        focals = focals.detach().cpu().numpy().astype(np.float32)
                        Ks = []
                        cx = float(W // 2)
                        cy = float(H // 2)
                        for fb in focals:
                            Ks.append(np.array([[fb, 0.0, cx], [0.0, fb, cy], [0.0, 0.0, 1.0]], dtype=np.float32))
            except Exception:
                Ks = None
            if Ks is None:
                K = self._generate_pseudo_intrinsics(H, W)
                Ks = [K for _ in range(B)]
            ray_maps = [self._ray_map_from_pose(c2w_mats[b], H, W, Ks[b]) for b in range(B)]
            ray_maps = torch.from_numpy(np.stack(ray_maps, axis=0)).float().to(device)  # (B,H,W,6)
            true_shape = torch.from_numpy(np.int32(np.tile([H, W], (B, 1)))).to(device)  # (B,2)

            # 2) encode ray-only input
            ray_maps_chw = ray_maps.permute(0, 3, 1, 2)  # (B,6,H,W)
            ray_out, ray_pos, _ = self._encode_ray_map(ray_maps_chw, true_shape)

            # NEW: use ray features to re-inquire pose token (no mem update)
            if self.pose_head_flag:
                global_ray_feat = self._get_img_level_feat(ray_out[-1])  # (B,1,C)
                pose_feat_i_ray = self.pose_retriever.inquire(global_ray_feat, mem)
                pose_pos_i_ray = -torch.ones(
                    pose_feat_i_ray.shape[0], 1, 2, device=pose_feat_i_ray.device, dtype=ray_pos.dtype
                )
            else:
                pose_feat_i_ray, pose_pos_i_ray = None, None

            # 3) rollout with OLD state (no img_mask/reset/update)
            new_state_feat_ray, dec_ray, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                ray_out[-1],
                ray_pos,
                pose_feat_i_ray,
                pose_pos_i_ray,
                init_state_feat,
                img_mask=None,
                reset_mask=None,
                update=None,
                return_attn=True,
            )

            # Also run downstream head on ray-only decoder outputs to get a full static prediction
            assert len(dec_ray) == self.dec_depth + 1
            head_input_static = [
                dec_ray[0].float(),
                dec_ray[self.dec_depth * 2 // 4][:, 1:].float(),
                dec_ray[self.dec_depth * 3 // 4][:, 1:].float(),
                dec_ray[self.dec_depth].float(),
            ]
            res_static = self._downstream_head(head_input_static, true_shape, pos=ray_pos)

            # Add per-point RGB from the original input view for static branch
            try:
                if (
                    isinstance(res_static, dict)
                    and ("pts3d_in_self_view" in res_static)
                    and ("img" in view)
                ):
                    # Normalize input image from [-1,1] -> [0,1]
                    img = view["img"]  # expected (B,3,H,W)
                    B, H, W, _ = res_static["pts3d_in_self_view"].shape

                    # Ensure image is Bx3xHxW then resize if needed to match (H,W)
                    if img.ndim == 4 and img.shape[1] in (3, 4):
                        img_chw = img[:, :3]
                    elif img.ndim == 4 and img.shape[-1] in (3, 4):
                        img_chw = img[..., :3].permute(0, 3, 1, 2)
                    else:
                        img_chw = img

                    if img_chw.shape[-2:] != (H, W):
                        img_chw = F.interpolate(
                            img_chw, size=(H, W), mode="bilinear", align_corners=False
                        )

                    rgb01 = 0.5 * (img_chw + 1.0)  # (B,3,H,W) in [0,1]
                    rgb_hwc = rgb01.permute(0, 2, 3, 1).contiguous()  # (B,H,W,3)
                    res_static["rgb_in_self_view"] = rgb_hwc.view(B, H * W, 3)
            except Exception:
                # colorizing static output is best-effort; ignore on failure
                pass

            out.update(
                state_feat_static=new_state_feat_ray,
                dec_ray=dec_ray,
                res_static=res_static,
                self_attn_state=self_attn_state,
                cross_attn_state=cross_attn_state,
                self_attn_img=self_attn_img,
                cross_attn_img=cross_attn_img,
                ray_pos=ray_pos,
            )
            return out
        except Exception:
            return out

    @torch.no_grad()
    def _compute_alpha_state_from_static(
        self,
        res_main: dict,
        res_static: dict,
        cross_attn_state,
        tau: float = 0.03,
        gamma: float = 3.0,
        return_img_map: bool = False,
        gamma_img: float = 1.5,
        floor_img: float = 0.15,
        mode: str = "depth",
    ) -> Optional[torch.Tensor]:
        """Compute per-state-token gate alpha_state ∈ [0,1] from main vs static outputs.

        - res_main/res_static must contain 'pts3d_in_self_view' (B,H,W,3).
        - cross_attn_state: iterable of tensors with shape (num_heads, N_state, N_img).
        - patch_size: image patch size corresponding to one image token.
        - mode: 'depth' uses |z_main - z_static|; 'l1-3d' uses L1 distance of 3D points.

        Returns alpha_state with shape (B, N_state, 1), or None if unavailable.
        """
        try:
            if ("pts3d_in_self_view" not in res_main) or ("pts3d_in_self_view" not in res_static):
                return None
            p_main = res_main["pts3d_in_self_view"]  # (B,H,W,3)
            p_stat = res_static["pts3d_in_self_view"]
            if p_main.ndim != 4 or p_stat.ndim != 4:
                return None
            assert p_main.shape[:3] == p_stat.shape[:3]

            # Filter static branch predictions by confidence threshold
            conf_threshold = 0.1
            if "conf_self" in res_static:
                conf_stat = res_static["conf_self"]
                if conf_stat.ndim == 4:
                    conf_stat = conf_stat[..., 0]  # (B,H,W)
                # Create mask for confident predictions
                conf_mask = (conf_stat >= conf_threshold).float()  # (B,H,W)
            else:
                conf_mask = torch.ones_like(p_stat[..., 0])  # (B,H,W)

            if mode == "l1-3d":
                delta = (p_main - p_stat).abs().sum(dim=-1)  # (B,H,W)
                # relative normalization to reduce far-range dominance
                denom = p_main.norm(dim=-1).clamp_min(1e-6)
                delta = delta / denom
            else:  # 'depth'
                delta = (p_main[..., 2] - p_stat[..., 2]).abs()  # (B,H,W)
                denom = p_main[..., 2].abs().clamp_min(1e-6)
                delta = delta / denom

            # Apply confidence mask: zero out low-confidence regions
            delta = delta * conf_mask
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            B, H, W = delta.shape

            # build a pixel-level alpha_img first (optional return)
            if return_img_map:
                try:
                    flat = delta.view(B, -1)
                    q25 = torch.quantile(flat, 0.25, dim=1, keepdim=True)
                    q50 = torch.quantile(flat, 0.50, dim=1, keepdim=True)
                    q75 = torch.quantile(flat, 0.75, dim=1, keepdim=True)
                    tau_img = q50
                    iqr_img = (q75 - q25).clamp_min(1e-6)
                    scale_img = gamma_img / iqr_img
                except Exception:
                    tau_img = torch.full((B, 1), tau, device=delta.device, dtype=delta.dtype)
                    scale_img = torch.full((B, 1), gamma_img, device=delta.device, dtype=delta.dtype)
                alpha_img = torch.sigmoid(scale_img.view(B, 1, 1) * (tau_img.view(B, 1, 1) - delta)).clamp(0.0, 1.0)

                # optional confidence modulation if available
                if ("conf_self" in res_main) and ("conf_self" in res_static):
                    wmap = torch.minimum(res_main["conf_self"][..., 0] if res_main["conf_self"].ndim == 4 else res_main["conf_self"],
                                         res_static["conf_self"][..., 0] if res_static["conf_self"].ndim == 4 else res_static["conf_self"])  # (B,H,W)
                    wmap = torch.log(wmap.clamp_min(1e-6))
                    wmap = torch.nan_to_num(wmap, nan=0.0)
                    wv = wmap.view(B, -1)
                    wmin = wv.min(dim=1, keepdim=True).values
                    wmax = wv.max(dim=1, keepdim=True).values
                    wnorm = ((wmap - wmin.view(B, 1, 1)) / (wmax - wmin + 1e-6).view(B, 1, 1)).clamp(0.0, 1.0)
                    alpha_img = (alpha_img * (0.5 + 0.5 * wnorm)).clamp(0.0, 1.0)

                # smooth + floor + channelize
                alpha_img = F.avg_pool2d(alpha_img.unsqueeze(1), 3, 1, 1)
                alpha_img = floor_img + (1.0 - floor_img) * alpha_img.clamp(0.0, 1.0)
                alpha_img = alpha_img.clamp(0.0, 1.0)
                alpha_img = alpha_img.detach()  # safety

            # aggregate attention across layers/heads, remove pose key, keep batch if present
            if not (isinstance(cross_attn_state, (list, tuple)) and len(cross_attn_state) > 0):
                return None

            A_list = []  # will hold either (B,N_state,N_img) per layer or (N_state,N_img)
            has_batch = False
            for att in cross_attn_state:
                # att could be (num_heads, N_state, 1+N_img) or (B, num_heads, N_state, 1+N_img)
                if att.ndim == 4:
                    # assume (B,H,N_state,1+N_img) and values are logits (pre-softmax)
                    has_batch = True
                    att = F.softmax(att, dim=-1)
                    att = att[..., 1:]  # drop pose key
                    # re-normalize after dropping pose key (numerical safety)
                    att = att / att.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    att = att.mean(dim=1)  # mean over heads -> (B,N_state,N_img)
                    A_list.append(att)
                elif att.ndim == 3:
                    # (H,N_state,1+N_img)
                    att = F.softmax(att, dim=-1)
                    att = att[..., 1:]  # drop pose key
                    att = att / att.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    att = att.mean(dim=0)  # -> (N_state,N_img)
                    A_list.append(att)
                else:
                    continue

            if len(A_list) == 0:
                return None

            if has_batch:
                # stack over layers and mean: result (B,N_state,N_img)
                # ensure all have batch dim
                A = torch.stack(A_list, dim=0).mean(dim=0)
            else:
                # (L, N_state, N_img) -> (N_state,N_img), then broadcast to B
                A = torch.stack(A_list, dim=0).mean(dim=0)
                A = A[None].expand(B, -1, -1)

            A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)  # (B,N_state,N_img)
            N_img = A.shape[-1]

            # build confidence weights and do weighted pooling to N_img
            if ("conf_self" in res_main) and ("conf_self" in res_static):
                w = torch.minimum(res_main["conf_self"], res_static["conf_self"])  # (B,*,*)
                if w.ndim == 2:
                    w = w.view(B, H, W)
                # log-scale + per-sample min-max to [0,1]
                w = torch.log(w.clamp_min(1e-6))
                w = torch.nan_to_num(w, nan=0.0)
                # per-sample min-max normalized (main-branch) confidence weighting
                w = w.view(B, -1)
                w_min = w.min(dim=1, keepdim=True).values
                w_max = w.max(dim=1, keepdim=True).values
                w = (w - w_min) / (w_max - w_min + 1e-6)
                w = w.clamp(0.0, 1.0)
            else:
                w = torch.ones(B, H * W, device=delta.device, dtype=delta.dtype)

            delta_flat = delta.view(B, -1)
            num = F.adaptive_avg_pool1d((w * delta_flat).unsqueeze(1), N_img).squeeze(1)
            den = F.adaptive_avg_pool1d(w.unsqueeze(1), N_img).squeeze(1)
            delta_tok = (num / den.clamp_min(1e-6))  # (B,N_img)

            # stash a p90 object-motion score (depth-relative branch discrepancy over image tokens)
            try:
                self._last_dyn = float(torch.quantile(delta_tok.detach().float().flatten(), 0.9).item())
            except Exception:
                self._last_dyn = 0.0

            # normalized projection per state (keys normalization)
            A_sum = A.sum(dim=-1).clamp_min(1e-6)  # (B,N_state)
            proj = torch.einsum("bsi,bi->bs", A, delta_tok) / A_sum  # (B,N_state)

            # adaptive threshold per batch using median/IQR
            try:
                q25 = torch.quantile(proj, 0.25, dim=1, keepdim=True)
                q50 = torch.quantile(proj, 0.50, dim=1, keepdim=True)
                q75 = torch.quantile(proj, 0.75, dim=1, keepdim=True)
                tau_b = q50
                # IQR floor for numerical safety (on near-static scenes IQR->0 would blow up scale)
                iqr = (q75 - q25).clamp_min(1e-6)
                scale = gamma / iqr
            except Exception:
                tau_b = torch.full_like(proj[:, :1], tau)
                scale = torch.full_like(proj[:, :1], gamma)

            alpha_state = torch.sigmoid(scale * (tau_b - proj)).clamp(0.0, 1.0)
            if return_img_map:
                return alpha_state[..., None], alpha_img  # (B,N_state,1), (B,1,H,W)
            return alpha_state[..., None]  # (B,N_state,1)
        except Exception as _e:
            # Log once so a disabled alpha gate is visible during eval instead of failing silently.
            if not getattr(self, "_alpha_fail_warned", False):
                import traceback, sys
                print(f"[raymap3r][WARN] alpha-state gating disabled this frame: {_e}", file=sys.stderr)
                traceback.print_exc()
                self._alpha_fail_warned = True
            return None

    @torch.no_grad()
    def _apply_alpha_gate(self, update_mask1: torch.Tensor, res: dict, view: dict, i: int) -> torch.Tensor:
        """Apply per-state gating derived from static consistency with EMA + coverage-aware scaling.

        - Reads `alpha_state` from `res` (shape (B,N_state,1), in [0,1]).
        - Maintains an EMA buffer `self._alpha_ema` across frames, reset on first frame or view reset.
        - Applies gating with floor and coverage-adaptive strength, plus small-step factor under large coverage.

        Args:
            update_mask1: base update mask, broadcastable to (B,N_state,1)
            res: dict containing optional key `alpha_state`
            view: current view dict containing optional `reset`
            i: frame index (int)

        Returns:
            update_mask1 after gating
        """
        alpha_state = res.get("alpha_state", None)
        if alpha_state is None:
            return update_mask1

        # temporal EMA smoothing of alpha_state
        reset_flag = view.get("reset", None)
        do_reset = False
        if i == 0:
            do_reset = True
        else:
            if isinstance(reset_flag, (bool, int)):
                do_reset = bool(reset_flag)
            elif torch.is_tensor(reset_flag):
                try:
                    do_reset = bool((reset_flag != 0).any().item())
                except Exception:
                    do_reset = False

        if do_reset or (self._alpha_ema is None):
            self._alpha_ema = alpha_state.detach()
        else:
            beta = 1.0 / max(self.alpha_ema_tau, 1.0)
            self._alpha_ema = (
                (1.0 - beta) * self._alpha_ema + beta * alpha_state
            ).detach()

        alpha = self._alpha_ema.to(update_mask1.dtype)
        # coverage-adaptive gating strength
        cover = (1.0 - alpha).mean(dim=1, keepdim=True)  # (B,1,1)
        lam = self.alpha_gate_lambda * (1.0 - self.coverage_adapt_k * cover)
        lam = lam.clamp(0.0, 1.0)
        # gate with floor
        floor = torch.full_like(alpha, self.alpha_gate_wmin)
        gate = torch.maximum(floor, (1.0 - lam) + lam * alpha)
        # small-step factor under large coverage
        eta = (1.0 - self.small_step_c * cover).clamp(0.5, 1.0)
        return update_mask1 * gate * eta

    # tbptt training encoder: Truncated Backpropagation Through Time
    def _forward_encoder(self, views):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        return (feat, pos, shape), (
            init_state_feat,
            init_mem,
            state_feat,
            state_pos,
            mem,
        )

    # tbptt training decoder step: Truncated Backpropagation Through Time
    def _forward_decoder_step(
        self,
        views,
        i,
        feat_i,
        pos_i,
        shape_i,
        init_state_feat,
        init_mem,
        state_feat,
        state_pos,
        mem,
    ):

        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=views[i]["img_mask"],
            reset_mask=views[i]["reset"],
            update=views[i].get("update", None),
            return_attn=False,
        )
        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape_i, pos=pos_i)
        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)
        if update is not None:
            update_mask = img_mask & update  # if don't update, then whatever img_mask
        else:
            update_mask = img_mask
        update_mask = update_mask[:, None, None].float()
        state_feat = new_state_feat * update_mask + state_feat * (
            1 - update_mask
        )  # update global state
        mem = new_mem * update_mask + mem * (1 - update_mask)  # then update local state
        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)
        return res, (state_feat, mem)

    # training and testing
    def _forward_impl(self, views, ret_state=False):
        # [B, C, H, W] -> [B, H/16*W/16, 1024]
        shape, feat_ls, pos = self._encode_views(views) # [15, 3, 288, 512] -> feat [15, 576, 1024], pos [15, 576, 2]
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0]) # init state feat [1, 768, 768], state_pos [1, 768, 2]
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1) # [1, 256, 1536] init pose mem
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i) # avg pool: [1, 576, 1024] -> [1, 1, 1024]
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1) # [1, 1, 768] init pose token
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem) 
                    # query mem with [global_img_feat, pose token] -> pose_feat_i
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                ) # [1, 1, 2]
            else:
                pose_feat_i = None
                pose_pos_i = None
            current_img_mask = views[i]["img_mask"]
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=current_img_mask,
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
                return_attn=True,
            )
            out_pose_feat_i = dec[-1][:, 0:1] # [1, 1, 768] refined pose token from dust3r
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            ) # [1, 256, 1536] use mem as query, cross-attend [global_img_feat_i, out_pose_feat_i], get new_mem
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(), # [1, 576, 1024]
                dec[self.dec_depth * 2 // 4][:, 1:].float(), # [1, 576, 768]
                dec[self.dec_depth * 3 // 4][:, 1:].float(), # [1, 576, 768]
                dec[self.dec_depth].float(), # [1, 1 + 576, 768]
            ]
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()

            # update with learning rate
            if i  == 0:
                update_mask1 = update_mask
            else:
                if self.config.model_update_type == "cut3r":
                    update_mask1 = update_mask
                elif self.config.model_update_type == "xattn":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)') # [12, 16, 768, 1 + 576] -> [1, 768, 1 + 576, 12*16]
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None] * 1.0
                else:
                    raise ValueError(f"Invalid model type: {self.config.model_update_type}")

            update_mask2 = update_mask
            state_feat = new_state_feat * update_mask1 + state_feat * (
                1 - update_mask1
            )  # update global state
            mem = new_mem * update_mask2 + mem * (
                1 - update_mask2
            )  # then update local state
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward(self, views, ret_state=False):
        if ret_state:
            ress, views, state_args = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            ress, views = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    # testing: generate rgb xyz condition on raymap
    def inference_step(
        self, view, state_feat, state_pos, init_state_feat, mem, init_mem
    ):
        batch_size = view["img"].shape[0]
        raymaps = []
        shapes = []
        for j in range(batch_size):
            assert view["ray_mask"][j]
            raymap = view["ray_map"][[j]].permute(0, 3, 1, 2)
            raymaps.append(raymap)
            shapes.append(
                view.get(
                    "true_shape",
                    torch.tensor(view["ray_map"].shape[-2:])[None].repeat(
                        view["ray_map"].shape[0], 1
                    ),
                )[[j]]
            )

        raymaps = torch.cat(raymaps, dim=0)
        shape = torch.cat(shapes, dim=0).to(raymaps.device)
        feat_ls, pos, _ = self._encode_ray_map(raymaps, shapes) # [1, 6, 384, 512] -> feat [1, 768, 1024], pos [1, 768, 2]

        feat_i = feat_ls[-1]
        pos_i = pos
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
            return_attn=False,
        )

        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        assert len(dec) == self.dec_depth + 1
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape, pos=pos_i)
        return res, view

    # recurrent testing
    def forward_recurrent(self, views, device, ret_state=False):
        ress = []
        all_state_args = []
        for i, view in enumerate(views):
            device = view["img"].device
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            ray_mask = view["ray_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = view["ray_map"].unsqueeze(
                0
            )  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(
                -1, *ray_maps.shape[2:]
            )  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(
                imgs.device
            )  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()
                all_state_args.append(
                    (state_feat, state_pos, init_state_feat, mem, init_mem)
                )

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                return_attn=False,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)
            ress.append(res)
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward_recurrent_lighter(self, views, device='cuda', ret_state=False):
        ress = []
        render_views = []  # align with ress length for downstream visualization
        all_state_args = []
        reset_mask = False
        warmup_frames = 5  # run main branch only for first N frames

        # State-aware trajectory smoothing (Sec 3.5): exp-smooths the output translation with
        # beta = 1/(1+gamma*|accel*state_change|), gated per-frame by the router and written onto the pose.
        _pose_buffer = []                      # list of (B,4,4) c2w matrices
        _inner_iax_delta_buffer = []           # raw translation deltas (for acceleration)
        _inner_iax_smoothed_delta = None       # running smoothed delta
        _inner_iax_smoothed_pos = None         # running smoothed position

        # Sec 3.4 Reset Metric Alignment: every _reset_every frames the state resets; a confidence-
        # weighted Umeyama Sim(3) from the repeated frame's pre/post-reset clouds corrects later poses.
        _reset_every = 100
        _reset_icp = True
        _M_accum = None  # (s, R(3x3), t(3)) mapping current-segment world -> anchor (segment-0) world
        def _wumeyama(src, dst, w):  # least-squares Sim3 mapping src->dst, weights w
            w = w / (w.sum() + 1e-9)
            mu_s = (w[:, None] * src).sum(0); mu_d = (w[:, None] * dst).sum(0)
            sc = src - mu_s; dc = dst - mu_d
            Sig = (w[:, None] * dc).T @ sc
            U, Dg, Vt = np.linalg.svd(Sig)
            Rm = U @ Vt
            if np.linalg.det(Rm) < 0:
                U[:, -1] *= -1; Rm = U @ Vt
            var_s = (w * (sc ** 2).sum(1)).sum()
            s = float(Dg.sum() / (var_s + 1e-9))
            t = mu_d - s * Rm @ mu_s
            return s, Rm, t
        def _sim3_apply_c2w(M, c2w):  # apply (s,R,t) to a (4,4) c2w; keeps rotation orthonormal
            s, Rm, t = M
            out = c2w.copy()
            out[:3, :3] = Rm @ c2w[:3, :3]
            out[:3, 3] = s * (Rm @ c2w[:3, 3]) + t
            return out

        # Adaptive rotation router (Sec 3): a short warm-up measures camera rotation and fixes the
        # regime once -- low rotation keeps CUT3R; object motion -> gated (xattn) update + smoothing.
        self._last_dyn = 0.0
        _router_W = 20            # warm-up window (frames) used to estimate rotation
        _router_gamma_dyn = 2.0   # smoothing-gate strength in the dynamic regime
        _router_tau_rot = 2.0     # rotation threshold (deg/frame) separating the regimes
        self._router_S = False
        self._router_decided = False
        self._router_rot = []
        self._router_prev_c2w = None
        self.config.model_update_type = "cut3r"  # warm-up base: unconditional CUT3R update
        # --- [george] forced-regime patch begin: bypass the adaptive router for A/B runs.
        # Set via `model.force_update_type` ("auto"|"cut3r"|"xattn", see infer.py flag).
        # "xattn"  -> gated regime fixed from frame 0 (alpha gate + smoothing active).
        # "cut3r"  -> TRUE vanilla baseline: cut3r update rule AND alpha gate disabled
        #             (a router-decided cut3r regime would still apply _apply_alpha_gate,
        #             so regime forcing alone is NOT a clean gate-off baseline).
        self.last_router_info = None
        self._force_no_alpha = False
        _force = str(getattr(self, "force_update_type", "auto"))
        if _force in ("cut3r", "xattn"):
            self.config.model_update_type = _force
            self._router_S = (_force == "xattn")
            self._router_decided = True
            self._force_no_alpha = (_force == "cut3r")
            self.last_router_info = {"forced": _force, "regime": _force,
                                     "median_rot_deg": None, "decided_at_frame": 0}
            print(f"[router] forced regime={_force} (adaptive router bypassed)")
        # --- [george] forced-regime patch end ---
        def _sig(x):
            return float(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))
        for i, _view in enumerate(views):
            view = to_gpu(_view, device)
            device = view["img"].device
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            ray_mask = view["ray_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = view["ray_map"].unsqueeze(
                0
            )  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(
                -1, *ray_maps.shape[2:]
            )  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(
                imgs.device
            )  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0 or reset_mask:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                return_attn=True,
            )
            out_pose_feat_i = dec[-1][:, 0:1]

            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)

            # apply accumulated reset-alignment Sim(3) to the OUTPUT pose of this segment
            if _reset_icp and (_M_accum is not None) and isinstance(res, dict) and ("camera_pose" in res):
                try:
                    _cw = pose_encoding_to_camera(res["camera_pose"]).detach().cpu().numpy()  # (B,4,4)
                    _cw_c = np.stack([_sim3_apply_c2w(_M_accum, _cw[b]) for b in range(_cw.shape[0])], 0)
                    res["camera_pose"] = camera_to_pose_encoding(
                        torch.from_numpy(_cw_c).to(res["camera_pose"].device, res["camera_pose"].dtype)
                    )
                except Exception as _e:
                    import sys; print(f"[raymap3r][WARN] reset-ICP apply frame {i}: {_e}", file=sys.stderr)

            # State-aware smoothing (Sec 3.5): engaged only in the dynamic regime after the router
            # has decided. Smooths the output translation with an acceleration*state-change gate.
            _override_c2w = None
            _use_inner = bool(self._router_S and self._router_decided)
            if _use_inner:
                _c2w_cur = pose_encoding_to_camera(res["camera_pose"]).detach().cpu().numpy()
                # clear buffers on reset
                _reset_flag = view.get("reset", False)
                _do_reset = False
                if isinstance(_reset_flag, (bool, int)):
                    _do_reset = bool(_reset_flag)
                elif torch.is_tensor(_reset_flag):
                    _do_reset = bool((_reset_flag != 0).any().item())
                if _do_reset:
                    _pose_buffer.clear()
                    _inner_iax_delta_buffer.clear()
                    _inner_iax_smoothed_delta = None
                    _inner_iax_smoothed_pos = None

                # raw translation delta vs previous frame
                if len(_pose_buffer) > 0:
                    _delta = _c2w_cur[:, :3, 3] - _pose_buffer[-1][:, :3, 3]
                else:
                    _delta = np.zeros_like(_c2w_cur[:, :3, 3])

                # acceleration = ||delta_t - delta_{t-1}||
                if len(_inner_iax_delta_buffer) > 0:
                    _accel = float(np.linalg.norm(_delta - _inner_iax_delta_buffer[-1], axis=-1).mean())
                else:
                    _accel = 0.0

                # state change between consecutive recurrent states
                with torch.no_grad():
                    _sc = float((new_state_feat - state_feat).norm(dim=-1).mean().item())

                # beta = 1 / (1 + gamma * |accel * sc|): near-static -> beta~1 (near identity),
                # high dynamic -> stronger smoothing.
                _beta = 1.0 / (1.0 + _router_gamma_dyn * abs(_accel * _sc))

                _pose_buffer.append(_c2w_cur)
                _inner_iax_delta_buffer.append(_delta)
                if _inner_iax_smoothed_delta is None:
                    _inner_iax_smoothed_delta = _delta.copy()
                    _inner_iax_smoothed_pos = _c2w_cur[:, :3, 3].copy()
                else:
                    _inner_iax_smoothed_delta = _beta * _delta + (1.0 - _beta) * _inner_iax_smoothed_delta
                    _inner_iax_smoothed_pos = _inner_iax_smoothed_pos + _inner_iax_smoothed_delta

                _override_c2w = _c2w_cur.copy()
                _override_c2w[:, :3, 3] = _inner_iax_smoothed_pos

            # write the smoothed translation back onto the OUTPUT pose so the reported ATE
            # reflects it (Sec 3.5). camera_pose enc is absT_quaR -> [:, :3] is translation.
            _do_S = bool(self._router_S and self._router_decided)
            if (_inner_iax_smoothed_pos is not None) and isinstance(res, dict) and ("camera_pose" in res) \
                    and _do_S:
                try:
                    _sm = torch.as_tensor(
                        _inner_iax_smoothed_pos,
                        device=res["camera_pose"].device,
                        dtype=res["camera_pose"].dtype,
                    )
                    res["camera_pose"] = res["camera_pose"].clone()
                    res["camera_pose"][:, :3] = _sm
                except Exception as _e:
                    import sys
                    print(f"[raymap3r][WARN] output-smooth skip frame {i}: {_e}", file=sys.stderr)

            if i < warmup_frames:
                # update mem with main-branch pose token (normalize for stability)
                new_mem = self.pose_retriever.update_mem(
                    mem, global_img_feat_i, out_pose_feat_i
                )
            else:
                # After obtaining res, run packed ray-only pipeline (ignore returns for now)
                static_out = self._remap_raymap(
                    res,
                    view,
                    state_feat,
                    state_pos,
                    init_state_feat,
                    mem,
                    device,
                    override_c2w=_override_c2w,
                )
                # compute per-state-token dynamic gate (and image map) from main vs static
                # Adjusted parameters for more sensitive dynamic detection:
                # - floor_img=0.0: allow alpha to go to 0 for highly dynamic regions
                # - gamma_img=2.5: increase sensitivity to differences
                alpha_state, alpha_img = self._compute_alpha_state_from_static(
                    res, static_out["res_static"], cross_attn_state,
                    return_img_map=True,
                    floor_img=0.0,  # allow full range [0, 1] instead of [0.15, 1]
                    gamma_img=2.5   # increase sensitivity (was 1.5)
                )
                res["alpha_state"] = alpha_state
                res["alpha_img"] = alpha_img
                # During the warm-up window, accumulate per-frame rotation; at i>=W decide the
                # sequence regime ONCE and fix it for the rest of the stream.
                if not self._router_decided:
                    try:
                        _c2w_r = pose_encoding_to_camera(res["camera_pose"]).detach().cpu().numpy()
                        _c2w_r = _c2w_r[0] if _c2w_r.ndim == 3 else _c2w_r
                        if self._router_prev_c2w is not None:
                            _Rr = _c2w_r[:3, :3] @ self._router_prev_c2w[:3, :3].T
                            _rdeg = float(np.degrees(np.arccos(np.clip((np.trace(_Rr) - 1.0) / 2.0, -1.0, 1.0))))
                        else:
                            _rdeg = 0.0
                        self._router_prev_c2w = _c2w_r
                        self._router_rot.append(float(_rdeg))
                        if i >= _router_W and len(self._router_rot) > 0:
                            _rotmed = float(np.median(self._router_rot))
                            _dyn = _rotmed < _router_tau_rot
                            if _dyn:
                                self.config.model_update_type = "xattn"; self._router_S = True
                            else:
                                self.config.model_update_type = "cut3r"; self._router_S = False
                            self._router_decided = True
                            # --- [george] record + log the router decision ---
                            self.last_router_info = {
                                "forced": "auto", "regime": self.config.model_update_type,
                                "median_rot_deg": round(_rotmed, 3), "decided_at_frame": i}
                            print(f"[router] decided regime={self.config.model_update_type} "
                                  f"(median_rot={_rotmed:.2f} deg/frame, frame {i})")
                            # --- [george] end ---
                    except Exception:
                        pass
                # update mem: static-branch pose token + static-weighted global key
                static_pose = static_out["dec_ray"][-1][:, 0:1]
                static_out_pose_feat_i = static_pose.detach()
                # base mean key (keeps distribution aligned with inquire)
                g0 = self._get_img_level_feat(feat_i)
                # static-weighted key from alpha_img
                gs = self._static_weighted_global_key_from_alpha_img(feat_i, alpha_img)
                # mild magnitude calibration to match g0 scale
                eps = 1e-6
                scale = (g0.norm(dim=-1, keepdim=True) + eps) / (gs.norm(dim=-1, keepdim=True) + eps)
                gs = gs * scale
                beta = float(self.static_key_mix_beta)
                beta = max(0.0, min(1.0, beta))
                global_k = (1.0 - beta) * g0 + beta * gs

                new_mem = self.pose_retriever.update_mem(
                    mem, global_img_feat_i, out_pose_feat_i
                )


            if i < warmup_frames:
                res_cpu = to_cpu(res)
                ress.append(res_cpu)
                render_views.append(view)
            else:
                res_cpu = to_cpu(res)
                ress.append(res_cpu)
                render_views.append(view)



                    
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()

            # update with learning rate (and optional alpha_state gating)
            if i  == 0 or reset_mask:
                update_mask1 = update_mask
            else:
                if self.config.model_update_type == "cut3r":
                    update_mask1 = update_mask
                elif self.config.model_update_type == "xattn":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)') # [12, 16, 768, 1 + 576] -> [1, 768, 1 + 576, 12*16]
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None] * 1.0
                else:
                    raise ValueError(f"Invalid model type: {self.config.model_update_type}")

            # apply per-state staticness gating (Sec 3.3), enabled once the router has decided
            # [george] _force_no_alpha: forced cut3r baseline disables the alpha gate entirely
            _do_alpha = bool(self._router_decided) and not getattr(self, "_force_no_alpha", False)
            if i > warmup_frames and _do_alpha:
                update_mask1 = self._apply_alpha_gate(update_mask1, res, view, i)

            update_mask2 = update_mask
            state_feat = new_state_feat * update_mask1 + state_feat * (
                1 - update_mask1
            )  # update global state
            mem = new_mem * update_mask2 + mem * (
                1 - update_mask2
            )  # then update local state

            reset_mask = view["reset"]
            # forced periodic reset + reset metric alignment (Sec 3.4)
            _force_reset = (_reset_every > 0 and i > 0 and (i % _reset_every == 0))
            if _reset_icp and _force_reset:
                try:
                    # pre-reset world points of the repeated frame (already M-corrected -> anchor coords)
                    _cw_pre = pose_encoding_to_camera(res["camera_pose"]).detach().cpu().numpy()[0]
                    _ps = res["pts3d_in_self_view"]
                    _ps = _ps[0] if _ps.ndim == 4 else _ps
                    _Ppre_self = _ps.detach().cpu().numpy().reshape(-1, 3)
                    # re-process the SAME frame with a FRESH state (post-reset prediction)
                    _pf = self.pose_token.expand(feat_i.shape[0], -1, -1) if self.pose_head_flag else None
                    _nsf2, _dec2, *_r2 = self._recurrent_rollout(
                        init_state_feat, state_pos, feat_i, pos_i, _pf, pose_pos_i, init_state_feat,
                        img_mask=view["img_mask"], reset_mask=torch.ones_like(view["reset"]),
                        update=None, return_attn=False,
                    )
                    _hi2 = [_dec2[0].float(), _dec2[self.dec_depth * 2 // 4][:, 1:].float(),
                            _dec2[self.dec_depth * 3 // 4][:, 1:].float(), _dec2[self.dec_depth].float()]
                    _res_post = self._downstream_head(_hi2, shape, pos=pos_i)
                    _cw_post = pose_encoding_to_camera(_res_post["camera_pose"]).detach().cpu().numpy()[0]
                    _qs = _res_post["pts3d_in_self_view"]
                    _qs = _qs[0] if _qs.ndim == 4 else _qs
                    _Ppost_self = _qs.detach().cpu().numpy().reshape(-1, 3)
                    # lift both to world: c2w (3x3) @ p + c
                    _Wpre = (_cw_pre[:3, :3] @ _Ppre_self.T).T + _cw_pre[:3, 3]
                    _Wpost = (_cw_post[:3, :3] @ _Ppost_self.T).T + _cw_post[:3, 3]
                    _cf = res.get("conf_self", None)
                    _w = _cf.detach().cpu().numpy().reshape(-1) if _cf is not None else np.ones(_Wpre.shape[0])
                    _n = min(_Wpre.shape[0], _Wpost.shape[0], _w.shape[0])
                    _Wpre, _Wpost, _w = _Wpre[:_n], _Wpost[:_n], _w[:_n]
                    if _n > 20000:
                        _idx = np.argsort(-_w)[:20000]
                        _Wpre, _Wpost, _w = _Wpre[_idx], _Wpost[_idx], _w[_idx]
                    # Umeyama mapping post(new-seg) -> pre(anchor): this IS the new accumulated correction
                    _M_accum = _wumeyama(_Wpost, _Wpre, _w)
                except Exception as _e:
                    import sys; print(f"[raymap3r][WARN] reset-ICP sim3 frame {i}: {_e}", file=sys.stderr)
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            if _force_reset:
                state_feat = init_state_feat.clone()
                mem = init_mem.clone()

        if ret_state:
            return ress, render_views, all_state_args
        return ress, render_views

if __name__ == "__main__":
    cfg = ARCroco3DStereoConfig(
        state_size=256,
        pos_embed="RoPE100",
        rgb_head=True,
        pose_head=True,
        img_size=(224, 224),
        head_type="linear",
        output_mode="pts3d+pose",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        pose_mode=("exp", -inf, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
    )
    ARCroco3DStereo(cfg)
