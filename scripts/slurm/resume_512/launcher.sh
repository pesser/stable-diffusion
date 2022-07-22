#!/bin/bash
export NODE_RANK=${SLURM_NODEID}
echo "##########################################"
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo NODE_RANK=${NODE_RANK}
echo WORLD_SIZE=${WORLD_SIZE}
echo "##########################################"
# debug environment worked great so we stick with it
# no magic there, just a miniconda python=3.9, pytorch=1.12, cudatoolkit=11.3
# env with pip dependencies from stable diffusion's requirements.txt
eval "$(/fsx/stable-diffusion/debug/miniconda3/bin/conda shell.bash hook)"
conda activate stable
cd /fsx/stable-diffusion/stable-diffusion

CONFIG=configs/stable-diffusion/txt2img-1p4B-multinode-clip-encoder-high-res-512.yaml

# initial parameters
#EXTRA="model.params.ckpt_path=/fsx/stable-diffusion/stable-diffusion/checkpoints/256f8ft512-2022-06-15-pruned.ckpt"

# resumed after crash
#EXTRA="model.params.ckpt_path=/fsx/stable-diffusion/stable-diffusion/logs/2022-07-06T23-43-51_txt2img-1p4B-multinode-clip-encoder-high-res-512/checkpoints/last.ckpt"

# continue on improved aesthetics
EXTRA="model.params.ckpt_path=/fsx/stable-diffusion/stable-diffusion/logs/2022-07-07T16-15-18_txt2img-1p4B-multinode-clip-encoder-high-res-512/checkpoints/last.ckpt data.params.tar_base=__improvedaesthetic__ -f _improvedaesthetic"

DEBUG="-d True lightning.callbacks.image_logger.params.batch_frequency=5"

python main.py --base $CONFIG --gpus 0,1,2,3,4,5,6,7 -t --num_nodes ${WORLD_SIZE} --scale_lr False $EXTRA #$DEBUG
