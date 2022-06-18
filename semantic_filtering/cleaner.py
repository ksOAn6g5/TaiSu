import torch
from models.model_infer import build_lit
from loader import TextImageDataset
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import os
import torch.distributed as dist
import random
import  torch.cuda.amp as amp
from utils.sp_tokenizer import SentencepieceChineseTokenizer
#image = preprocess(Image.open("test.jpg")).unsqueeze(0).to(device)

#os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,'
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
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
def check_consistency(lit,loader,args):
    ids = []
    cnt=0
    total=0
    for itr, data in enumerate(loader):
        key,text,image=data
        image_features=image.cuda(args.local_rank)
        text=text.cuda(args.local_rank)
        total+=image.shape[0]
        with torch.no_grad():
            # image_features = self.encode_image(image)
            with amp.autocast(enabled=True):
               text_features = lit.module.encode_text(text)
            # normalized features
               image_features = image_features / image_features.norm(dim=-1, keepdim=True)
               text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # cosine similarity as logits
            #logit_scale = lit..txt_model.logit_scale.exp()
               logits =image_features @ text_features.t()
           # logits= lit(image, text)
            size=120
            batch = len(logits) // size
            if not (len(logits) % size == 0):
                batch += 1
            for i in range(batch):
                tmp = logits[i * size:(i + 1) * size, i * size:(i + 1) * size]
                pred_x = torch.argmax(tmp, dim=1)
                pred_y = torch.argmax(tmp, dim=0)
                for j in range(len(tmp)):
                    if pred_x[j] == j or pred_y[j] == j:
                        ind = i * size + j
                        #print(tmp[j, j], key[ind])
                        cnt += 1
                        ids.append(key[ind])
        if itr%20==0:
            with open('./filtered_ofa/cleaned_ids{}.txt'.format(args.rank),'a',encoding='utf-8')as f:
                for ind in ids:
                    f.write(ind+'\n')
            ids=[]
        if args.rank==0:
           print('maiantaining rate:',cnt/total)
    with open('./filtered_ofa/cleaned_ids{}.txt'.format(args.rank),'a',encoding='utf-8')as f:
               for ind in ids:
                    f.write(ind+'\n')
    # if not os.patah.exists('./cleaned_keys'):
    #     os.makedirs('./cleaned_keys')

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--visual_model_path', type=str, required=True)
    parser.add_argument('--txt_model_path', type=str, required=True)
    parser.add_argument('--id2rank_lmdb', type=str, required=True)
    parser.add_argument('--img_lmdb_name', type=str, required=True)
    parser.add_argument('--text_lmdb_folder', type=str, required=True)
    parser.add_argument('--split', type=int, required=True)

    parser.add_argument('--batch_size', type=int, default=860)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist_url', type=str, help='distributed backend init_method')
    args = parser.parse_args()
###########################################################################
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
    set_seed(42 + rank)
    torch.cuda.set_device(args.local_rank)
    lit = build_lit(visual_model_path=args.visual_model_path, txt_model_path=args.txt_model_path)
    lit.cuda(args.local_rank)
    print('模型初始化完成')
    lit = torch.nn.SyncBatchNorm.convert_sync_batchnorm(lit)
    print('BN同步完成')
    lit = torch.nn.parallel.DistributedDataParallel(lit, device_ids=[args.local_rank],
                                                    output_device=args.local_rank)
    lit.eval()
    tokenizer = SentencepieceChineseTokenizer(52)
    dataset = TextImageDataset(split=1,
                 id2rank_lmdb=args.id2rank_lmdb,
                 img_lmdb_name=args.img_lmdb_name,
                 text_lmdb_folder=args.text_lmdb_folder,
                 tokenizer=tokenizer,
                 shuffle=True)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=args.rank
    )
    dataloder = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True, sampler=sampler,
                                            drop_last=False)
    print('start')
    check_consistency(lit=lit,loader=dataloder,args=args)
    print('finished')
    dist.destroy_process_group()  # 销毁进程组

