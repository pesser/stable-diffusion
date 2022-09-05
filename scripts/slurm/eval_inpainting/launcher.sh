#!/bin/bash

# mpi version for node rank
H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
export NODE_RANK=${THEID}
echo THEID=$THEID

echo "##########################################"
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo NODE_RANK=${NODE_RANK}
echo WORLD_SIZE=${WORLD_SIZE}
echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo SLURM_PROCID=${SLURM_PROCID}
echo "##########################################"
# debug environment worked great so we stick with it
# no magic there, just a miniconda python=3.9, pytorch=1.12, cudatoolkit=11.3
# env with pip dependencies from stable diffusion's requirements.txt
eval "$(/fsx/stable-diffusion/debug/miniconda3/bin/conda shell.bash hook)"
#conda activate stable
conda activate torch111
cd /fsx/stable-diffusion/stable-diffusion

#/bin/bash /fsx/stable-diffusion/stable-diffusion/scripts/test_gpu.sh

EXTRA="--indir /fsx/stable-diffusion/data/eval-inpainting/random_thick_512 --worldsize 8 --rank ${SLURM_PROCID}"
EXTRA="${EXTRA} --ckpt ${1} --outdir /fsx/stable-diffusion/stable-diffusion/inpainting-eval-results/${2}"

echo "Running ${EXTRA}"
cd /fsx/stable-diffusion/stable-diffusion/
python scripts/inpaint_sd.py ${EXTRA}
