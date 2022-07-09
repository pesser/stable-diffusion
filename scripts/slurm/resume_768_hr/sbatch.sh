#!/bin/bash
#SBATCH --partition=compute-od-gpu
#SBATCH --job-name=stable-diffusion-768cont-resumehr
#SBATCH --nodes=20
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x_%j.%n.out

# nccl / efa stuff
module load intelmpi
source /opt/intel/mpi/latest/env/vars.sh
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
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

# pytorch multinode vars
# node rank should be set in launcher script
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=11338
export WORLD_SIZE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${WORLD_SIZE}

srun --output=%x_%j.%n.out bash /fsx/stable-diffusion/stable-diffusion/scripts/slurm/resume_768_hr/launcher.sh  # srun vs mpirun?
