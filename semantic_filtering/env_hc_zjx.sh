module rm compiler/rocm/2.9
#module load apps/PyTorch/1.7-dynamic/hpcx-2.4.1-gcc-7.3.1-rocm3.9
#module load  apps/PyTorch/1.6.0a0/hpcx-2.4.1-gcc-7.3.1-rocm3.3
module load compiler/rocm/4.0.1

export NCCL_SOCKET_IFNAME=eno

#conda activate dalle_py36


export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export MIOPEN_DEBUG_CONV_WINOGRAD=0
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
#export NCCL_IB_HCA=mlx5_$1
export HSA_USERPTR_FOR_PAGED_MEM=0
#export GLOO_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export MIOPEN_SYSTEM_DB_PATH=/temp/pytorch-miopen-2.8  
export LD_LIBRARY_PATH=/public/software/apps/DeepLearning/PyTorch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/apps/DeepLearning/PyTorch/leveldb-1.22-build/lib64/:/public/software/apps/DeepLearning/PyTorch/lmdb-0.9.24-build/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/apps/DeepLearning/PyTorch/opencv-2.4.13.6-build/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/apps/DeepLearning/PyTorch/openblas-0.3.7-build/lib/:$LD_LIBRARY_PATH

