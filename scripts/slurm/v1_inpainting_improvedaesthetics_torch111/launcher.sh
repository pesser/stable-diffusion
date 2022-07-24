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
cd /fsx/robin/stable-diffusion/stable-diffusion

CONFIG="/fsx/robin/stable-diffusion/stable-diffusion/configs/stable-diffusion/inpainting/v1-finetune-for-inpainting-laion-iaesthe.yaml"

# debugging
#EXTRA="${EXTRA} -d True lightning.callbacks.image_logger.params.batch_frequency=50"

python main.py --base $CONFIG --gpus 0,1,2,3,4,5,6,7 -t --num_nodes ${WORLD_SIZE} --scale_lr False #$EXTRA
