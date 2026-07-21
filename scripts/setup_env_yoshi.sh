#!/bin/bash
# One-time environment setup for RayMap3R on yoshi (RTX 4090, CUDA 12.1 toolkit).
# Follows README: conda py3.10 + torch 2.1.1 cu121 + requirements + curope build.
# Note: yoshi previously had ~/anaconda3 (bashrc init block remains) but it was
# wiped in a reinstall; we install fresh miniconda and re-run conda init.
# Run:  cd ~/Warehouse/raymap3r_george && nohup bash scripts/setup_env_yoshi.sh > setup_env.log 2>&1 &
set -e
set -x

cd "$HOME/Warehouse/raymap3r_george"

# 1. user-space miniconda if absent
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    "$HOME/miniconda3/bin/conda" init bash   # replaces the stale anaconda3 block
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 2. env (idempotent); conda-forge only — the default anaconda channels require
# an interactive ToS acceptance (CondaToSNonInteractiveError) on fresh installs
conda env list | grep -q "^raymap3r " || \
    conda create -n raymap3r python=3.10 -y -c conda-forge --override-channels
conda activate raymap3r

# 3. torch cu121 first, then the rest
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install gdown matplotlib
# torch 2.1.x cpp_extension imports "from pkg_resources import packaging",
# which setuptools>=70 removed -> pin below 70 for the curope build
pip install "setuptools<70" wheel

# 4. build CroCo v2 RoPE CUDA extension against cuda-12.1
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
cd src/croco/models/curope
python setup.py build_ext --inplace
cd "$HOME/Warehouse/raymap3r_george"

# 5. CUT3R checkpoint (RayMap3R is training-free, reuses it)
mkdir -p weights
if [ ! -f weights/cut3r_512_dpt_4_64.pth ]; then
    # bare file ID works on every gdown version (6.x removed --fuzzy; URL from
    # https://github.com/CUT3R/CUT3R README, checkpoint cut3r_512_dpt_4_64.pth)
    gdown "1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD" -O weights/cut3r_512_dpt_4_64.pth
fi

# 6. sanity checks
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
ls -lh weights/

echo SETUP_DONE
