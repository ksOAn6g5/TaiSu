'''
author yulong-XJTU
'''
import torch
import torch.distributed as dist
from utils.sp_tokenizer import SentencepieceChineseTokenizer
from tokenizer import ChineseTokenizer
import argparse
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import yaml
import torch.nn.functional as F
from models.modified_model import CLIP
#import time
#import json
#import ddp_utils
import torch.cuda.amp as amp
from loader import TextImageDataset
from utils.custom_schedulers import get_cosine_schedule_with_warmup

def exists(val):
    return val is not None
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
       return
    dist.barrier()

def save_model(epoch, model,optimizer,lr_scheduler,scaler,dir, fname=None):
        save_obj = {
            # 'hparams': PrefixLM_configure,
            'epoch': epoch,
            'weights': model.module.state_dict(),
            'opt_state': optimizer.state_dict(),
            'scheduler_state': lr_scheduler.state_dict(),
            'scaler': scaler.state_dict(),
        }
        if fname is None:
            path = os.path.join(dir, 'lit' + str(epoch) + '.pt')
        else:
            path = os.path.join(dir, fname)
        torch.save(save_obj, path)
def num_training_steps(args,dataset_len):
        """Total training steps inferred from datamodule and devices."""
        dataset_size = dataset_len
        num_devices =args.world_size
        effective_batch_size = args.batch_size * num_devices
        return (dataset_size // effective_batch_size) *args.num_epochs


def configure_optimizers(args,model_name, model,dataset_len,isViT=False):
    isViT=True
    lr = {
        "RN50": 5e-4,
        "RN101": 5e-4,
        "RN50x4": 5e-4,
        "RN50x16": 4e-4,
        "RN50x64": 3.6e-4,
        "ViT-B/32": 5e-4,
        "ViT-B/16": 5e-4,
        "ViT-L/14": 4e-4,
        "ViT-L/14-336px": 2e-5
    }[model_name]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(
            0.9,
            0.999
        ),
        eps=1e-8,
        weight_decay=0.2
    )

    # Source: https://github.com/openai/CLIP/issues/107
    # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'


    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=num_training_steps(args,dataset_len))

    #yulonggaide
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
    #                                             num_training_steps=91000)#num_training_steps(args,dataset_len))


    return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def train_one_epoch(epoch,model,dataloader,data_sampler,optimizer,lr_scheduler,scaler,save_every_n_steps,rank,local_rank,checkpoint_dir):
    torch.cuda.empty_cache()
    data_sampler.set_epoch(epoch)
    for i, (txt, img_emb) in enumerate(dataloader):
        torch.cuda.empty_cache()
        txt=txt.cuda(local_rank)
        img_emb = img_emb.cuda(local_rank)
        with amp.autocast(enabled=not args.disable_amp):
             txt_emb = model.module.encode_text(txt)
             del txt
             img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
             txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

             logit_scale = model.module.logit_scale.exp()
             logits_per_image = logit_scale * img_emb @ txt_emb.t()

             labels = torch.arange(len(logits_per_image)).cuda(local_rank)
             image_loss = F.cross_entropy(logits_per_image, labels)
             text_loss = F.cross_entropy(logits_per_image.t(), labels)
             loss = (image_loss + text_loss) / 2
             model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

        scaler.scale(loss).backward()  # loss.backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        #     optimizer.step()
        lr_scheduler.step()
        if i%10==0 and rank==0:
            print('iter:{} ----------loss:{}'.format(i,loss.item()))
        if i%save_every_n_steps==0 and rank==0:
            save_model(epoch, model,optimizer,lr_scheduler,scaler,checkpoint_dir, fname=None)
    if rank==0:  save_model(epoch, model,optimizer,lr_scheduler,scaler,checkpoint_dir, fname=None)



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def main(args):
    #############################进程组#################
    rank = args.rank  # int(os.environ['RANK'])  #获取当前进程号
    # world_size=int(os.environ['WORLD_SIZE'])
    dist.init_process_group(
        backend='nccl',
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )  # 初始化
    assert dist.is_initialized()
    synchronize()
    print('进程组初始化完成')
    set_seed(42+rank)
    torch.cuda.set_device(args.local_rank)

    start_epoch = 0


    ################################# Resume ################

    RESUME = exists(args.resume)
    if RESUME:
        assert os.path.exists(args.resume), 'model file does not exist'
        loaded_obj = torch.load(args.resume, map_location='cpu')

        start_epoch, weights = loaded_obj['epoch'], loaded_obj['weights']
        opt_state = loaded_obj.get('opt_state')
        scheduler_state = loaded_obj.get('scheduler_state')
        scaler_state=loaded_obj.get('scaler')

    ############################dataset#####################
    if args.tokenizer=='wordpiece':
        tokenizer =ChineseTokenizer()
    else:
        tokenizer=SentencepieceChineseTokenizer(args.txt_seq_len)
    print('dataset 初始化')
    train_dataset = TextImageDataset(
        split=args.split,
		id2rank_lmdb=args.id2rank_lmdb,
        img_lmdb_name=args.img_lmdb_name,
        text_lmdb_folder=args.text_lmdb_folder,
        tokenizer=tokenizer,
        shuffle=True
    )


    dataset_len=len(train_dataset)
    print('loading dataset is complete!')
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank
    )
    print('dataloader 初始化')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=False
                                               )
    if 'RN' in args.model_name:
        config_file='./models/configs/RN.yaml'
    else:
        config_file = './models/configs/ViT.yaml'
    with open(config_file,'r',encoding='utf-8')as f:
         config=yaml.load(f)[args.model_name]
    model =CLIP(**config)
    if rank==0:
       print(config)
    if RESUME:
        model.load_state_dict(weights)
    model.cuda(args.local_rank)
    print('模型初始化完成')
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print('BN同步完成')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    print('DDP model')
    optimizers = configure_optimizers(args=args,model_name=args.model_name, model=model,
                                      dataset_len=dataset_len,isViT='ViT' in args.model_name)
    optimizer = optimizers['optimizer']
    lr_scheduler = optimizers['lr_scheduler']
    scaler = amp.GradScaler(enabled=not args.disable_amp)
    if RESUME:
        optimizer.load_state_dict(opt_state)
        lr_scheduler.load_state_dict(scheduler_state)
        scaler.load_state_dict(scaler_state)


    model.train()
    print('正在同步')
    synchronize()
    for epoch in range(start_epoch, args.num_epochs):
        train_one_epoch(epoch, model, dataloader=train_loader, data_sampler=train_sampler, optimizer=optimizer,
                        lr_scheduler=lr_scheduler,scaler=scaler, save_every_n_steps=args.save_every_n_steps, rank=rank,
                        local_rank=args.local_rank, checkpoint_dir=args.checkpoint_dir)
    dist.destroy_process_group()  # 销毁进程组


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--tokenizer', type=str, default='wordpiece')
    parser.add_argument('--split', type=int, help='0:all;1:web')
    parser.add_argument('--id2img_lmdb_json', type=str, help='id2img_lmdb_json')
    parser.add_argument('--id2img_lmdb', type=str, help='id2img_lmdb_json')
    parser.add_argument('--id2rank_lmdb',type=str,help='id2rank_lmdb')
    
    
    parser.add_argument('--img_lmdb_name', type=str, help='img_lmdb_name')
    parser.add_argument('--text_lmdb_folder', type=str, help='text_lmdb_folder')
    parser.add_argument('--txt_seq_len', type=int, default=52, help='max len of  texts')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist_url', type=str, help='distributed backend init_method')
    parser.add_argument('--num_workers', type=int, default=0, help='num of wokers for dataloader')
    parser.add_argument('--num_epochs', type=int, default=50, help='how many epochs to train')

    parser.add_argument('--shuffle', type=bool, default=False, help='whether permute the order of samples')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='path to your save checkpoint')
    parser.add_argument('--resume', type=str, default=None,help='path to your partially trained model')
    parser.add_argument('--save_every_n_steps', default=200, type=int, help='Save a checkpoint every n steps')
    parser.add_argument('--num_warmup_steps', default=2000, type=int, help='num_warmup_steps')
    parser.add_argument('--ids_file', type=str, help='lmdb_id')
    parser.add_argument('--ranks_file',type=str, help='lmdb_rank')
    parser.add_argument('--disable-amp', action='store_true',help='disable mixed-precision training (requires more memory and compute)')

    args = parser.parse_args()

    import time
    # time.sleep(600000)
    if args.rank==0:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
    main(args)
