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
conda activate torch111
cd /fsx/stable-diffusion/stable-diffusion

CONFIG="/fsx/stable-diffusion/stable-diffusion/configs/stable-diffusion/v1_improvedaesthetics.yaml"

# resume and set new seed to reshuffle data
#EXTRA="--seed 718 model.params.ckpt_path=/fsx/stable-diffusion/stable-diffusion/checkpoints2/v1pp/v1pp-flatline.ckpt"
#EXTRA="--seed 718 --resume_from_checkpoint /fsx/stable-diffusion/stable-diffusion/logs/2022-07-22T07-45-07_v1_improvedaesthetics/checkpoints/last.ckpt"
#EXTRA="--seed 719 --resume_from_checkpoint /fsx/stable-diffusion/stable-diffusion/logs/2022-07-22T12-32-32_v1_improvedaestheticsv1_iahr_torch111/checkpoints/last.ckpt"
#EXTRA="--seed 720 --resume_from_checkpoint /fsx/stable-diffusion/stable-diffusion/logs/2022-07-23T07-52-21_v1_improvedaestheticsv1_iahr_torch111/checkpoints/last.ckpt"
#EXTRA="--seed 721 --resume_from_checkpoint /fsx/stable-diffusion/stable-diffusion/logs/2022-07-24T19-07-33_v1_improvedaestheticsv1_iahr_torch111/checkpoints/last.ckpt"
EXTRA="--seed 722 --resume_from_checkpoint /fsx/stable-diffusion/stable-diffusion/logs/2022-07-29T10-26-01_v1_improvedaestheticsv1_iahr_torch111_ucg/checkpoints/last.ckpt"

# only images >= 512 and pwatermark <= 0.4999
EXTRA="${EXTRA} data.params.min_size=512 data.params.max_pwatermark=0.4999"

# unconditional guidance training
EXTRA="${EXTRA} model.params.ucg_training.txt.p=0.1 model.params.ucg_training.txt.val=''"

# reduce lr a bit
EXTRA="${EXTRA} model.params.scheduler_config.params.f_max=[0.5]"

# postfix
EXTRA="${EXTRA} -f v1_iahr_torch111_ucg"

# time to decay
#EXTRA="${EXTRA} model.params.scheduler_config.params.cycle_lengths=[300000] model.params.scheduler_config.params.warm_up_steps=[250000] model.params.scheduler_config.params.f_min=[1e-6]"

# custom logdir
#EXTRA="${EXTRA} --logdir rlogs"

# debugging
#EXTRA="${EXTRA} -d True lightning.callbacks.image_logger.params.batch_frequency=50"

/bin/bash /fsx/stable-diffusion/stable-diffusion/scripts/test_gpu.sh

python main.py --base $CONFIG --gpus 0,1,2,3,4,5,6,7 -t --num_nodes ${WORLD_SIZE} --scale_lr False $EXTRA
