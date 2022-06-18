#!/bin/bash
source env_hc.sh $OMPI_COMM_WORLD_LOCAL_RANK
echo $NCCL_SOCKET_IFNAME
lrank=$OMPI_COMM_WORLD_LOCAL_RANK
comm_rank=$OMPI_COMM_WORLD_RANK
comm_size=$OMPI_COMM_WORLD_SIZE
#/public/home/multimodel/anaconda3/envs/dalle_py36/bin
# --id2img_lmdb_json  meiyongshang 

#--resume=/work1/multimodel/AAA_LiT/ckpt_web_RN101_ep100_50nodes/lit48.pt \
####  return [int(176625970),int(110730104)][self.split]
###return [int(176625970),int(172941363)][self.split]
#[int(176625970),int(133071474)][self.split]   
#
# --text_lmdb_folder=/work1/multimodel/caption\     没有清洗过的
###    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
    ###                                            num_training_steps=91000)#num_training_steps(args,dataset_len))
    ##--resume=/work1/multimodel/AAA_LiT/ckpt_web_RN101_ep120_50nodes_bs860/lit119.pt \
    # 
    #/work1/multimodel/baidu_images_features/id2rank_RN_in_idcap
APP="/public/home/multimodel/anaconda3/envs/dalle_py36/bin/python3.6 train_clip.py --model_name=RN101 \
     --split=0 \
     --img_lmdb_name=/work1/multimodel/baidu_images_features/baidu_lmdb \
     --text_lmdb_folder=/public/home/multimodel/multimodel2/web_ofa_v2_str \
     --txt_seq_len=52 \
     --batch_size=860 \
     --num_workers=8 \
     --num_epochs=30  \
     --resume=./ckpt_web+ofa_txt_v2_120_epoch+word_tk/lit23.pt \
     --checkpoint_dir=./ckpt_web+ofa_txt_v2_120_epoch+word_tk  \
     --save_every_n_steps=200 \
     --num_warmup_steps=2000 \
     --dist_url tcp://${1}:43168 \
     --world_size=${comm_size} \
     --rank=${comm_rank} 
     --id2rank_lmdb=/public/home/multimodel/multimodel2/clean_web+clean_ofa_id_rank_RN  \
      "

     #       --ids_file=/public/home/actqrzwa6p/CLIP_zb/AAA_LiT/id_rank_json_id.txt\
     # --ranks_file=/public/home/actqrzwa6p/CLIP_zb/AAA_LiT/id_rank_json_rank.txt\
     # --id2img_lmdb_json=/public/home/actqrzwa6p/CLIP_zb/AAA_LiT/yulong/CLIP_embed/baidu_lmdb_rank_0.json\
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

