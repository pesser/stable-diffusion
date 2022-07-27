#!/bin/bash
#SBATCH --partition=compute-od-gpu
#SBATCH --job-name=stable-diffusion-v1-upscaling-f16-pretraining-512-aesthetics
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=%x_%j.out
#SBATCH --comment "Key=Monitoring,Value=ON"

module load intelmpi
source /opt/intel/mpi/latest/env/vars.sh
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-inst
all/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export NCCL_PROTO=simple
export PATH=/opt/amazon/efa/bin:$PATH
export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export NCCL_DEBUG=info
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export WORLD_SIZE=$COUNT_NODE

echo go $COUNT_NODE
echo $HOSTNAMES
echo $WORLD_SIZE

mpirun -n $COUNT_NODE -perhost 1 /fsx/robin/stable-diffusion/stable-diffusion/scripts/slurm/v1-upscaling-f16-pretraining-512-aesthetics/launcher.sh
