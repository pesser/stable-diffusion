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
conda activate torch111
cd /fsx/stable-diffusion/stable-diffusion

CONFIG="/fsx/stable-diffusion/stable-diffusion/configs/stable-diffusion/v1_laionhr.yaml"

# resume and set new seed to reshuffle data
EXTRA="--seed 718 model.params.ckpt_path=/fsx/stable-diffusion/stable-diffusion/checkpoints2/v1pp/v1pp-flatline.ckpt"

# time to decay
#EXTRA="${EXTRA} model.params.scheduler_config.params.cycle_lengths=[50000] model.params.scheduler_config.params.f_min=[1e-6]"

# custom logdir
#EXTRA="${EXTRA} --logdir rlogs"

# debugging
#EXTRA="${EXTRA} -d True lightning.callbacks.image_logger.params.batch_frequency=50"

python main.py --base $CONFIG --gpus 0,1,2,3,4,5,6,7 -t --num_nodes ${WORLD_SIZE} --scale_lr False $EXTRA
