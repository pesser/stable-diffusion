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
echo "##########################################"
# debug environment worked great so we stick with it
# no magic there, just a miniconda python=3.9, pytorch=1.12, cudatoolkit=11.3
# env with pip dependencies from stable diffusion's requirements.txt
eval "$(/fsx/stable-diffusion/debug/miniconda3/bin/conda shell.bash hook)"
#conda activate stable
# torch 1.11 to avoid bug in ckpt restoring
conda activate torch111
cd /fsx/stable-diffusion/stable-diffusion

CONFIG=configs/stable-diffusion/v3_pretraining.yaml

# resume and set new seed to reshuffle data
#EXTRA="--seed 714 model.params.ckpt_path=/fsx/stable-diffusion/stable-diffusion/rlogs/2022-07-11T22-57-10_txt2img-v2-clip-encoder-improved_aesthetics-256/checkpoints/last.ckpt"
#EXTRA="--seed 715 --resume_from_checkpoint /fsx/stable-diffusion/stable-diffusion/logs/2022-07-14T21-03-49_txt2img-v2-clip-encoder-improved_aesthetics-256/checkpoints/last.ckpt"
#EXTRA="--seed 716 --resume_from_checkpoint /fsx/stable-diffusion/stable-diffusion/logs/2022-07-22T09-25-26_v3_pretraining/checkpoints/last.ckpt"
#EXTRA="--seed 718 --resume_from_checkpoint /fsx/stable-diffusion/stable-diffusion/logs/2022-07-24T06-15-08_v3_pretraining/checkpoints/last.ckpt"
EXTRA="--seed 719 --resume_from_checkpoint /fsx/stable-diffusion/stable-diffusion/logs/2022-07-26T08-15-23_v3_pretraining/checkpoints/last.ckpt"

# reduce lr a bit
EXTRA="${EXTRA} model.params.scheduler_config.params.f_max=[0.75]"

# custom logdir
#EXTRA="${EXTRA} --logdir rlogs"

# debugging
#EXTRA="${EXTRA} -d True lightning.callbacks.image_logger.params.batch_frequency=50"

# detect bad gpus early on
/bin/bash /fsx/stable-diffusion/stable-diffusion/scripts/test_gpu.sh

python main.py --base $CONFIG --gpus 0,1,2,3,4,5,6,7 -t --num_nodes ${WORLD_SIZE} --scale_lr False $EXTRA
