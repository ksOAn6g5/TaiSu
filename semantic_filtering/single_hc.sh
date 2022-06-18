#!/bin/bash

source env_hc_zjx.sh $OMPI_COMM_WORLD_LOCAL_RANK
echo $NCCL_SOCKET_IFNAME

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
#

lrank=$OMPI_COMM_WORLD_LOCAL_RANK
comm_rank=$OMPI_COMM_WORLD_RANK
comm_size=$OMPI_COMM_WORLD_SIZE

img_lmdb_name=/work1/multimodel/baidu_images_features/baidu_lmdb  #填32个文件夹的公共路径+_rank之前的名字部分
text_lmdb_folder=/public/home/multimodel/multimodel2/merge_baidu_0522_lmdb
id2rank_lmdb=/public/home/multimodel/multimodel2/baidu_images_all_predict_tran_part_7_in_id2rank_new_data
APP="python3  cleaner.py \
        --visual_model_path=./models/RN101.pth \
        --txt_model_path=/work1/multimodel/AAA_LiT/ckpt_web_RN101_ep130_50nodes_bs860_resume_web1_true_2.2_1.7/lit129.pt  \
        --batch_size=720 \
        --num_workers=8 \
        --split=0 \
        --id2rank_lmdb=${id2rank_lmdb}  \
        --img_lmdb_name=${img_lmdb_name}  \
        --text_lmdb_folder=${text_lmdb_folder} \
        --dist_url tcp://${1}:43168 \
        --world_size=${comm_size} \
        --rank=${comm_rank} \
      "

case ${lrank} in
[0])
  export HIP_VISIBLE_DEVICES=0
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  echo numactl --cpunodebind=0 --membind=0 ${APP}
  numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[1])
  export HIP_VISIBLE_DEVICES=1
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  echo numactl --cpunodebind=1 --membind=1 ${APP}
  numactl --cpunodebind=1 --membind=1 ${APP}
  ;;
[2])
  export HIP_VISIBLE_DEVICES=2
  export UCX_NET_DEVICES=mlx5_2:1
  export UCX_IB_PCI_BW=mlx5_2:50Gbs
  echo numactl --cpunodebind=2 --membind=2 ${APP}
  numactl --cpunodebind=2 --membind=2 ${APP}
  ;;
[3])
  export HIP_VISIBLE_DEVICES=3
  export UCX_NET_DEVICES=mlx5_3:1
  export UCX_IB_PCI_BW=mlx5_3:50Gbs
  echo numactl --cpunodebind=3 --membind=3 ${APP}
  numactl --cpunodebind=3 --membind=3 ${APP}
  ;;
esac

