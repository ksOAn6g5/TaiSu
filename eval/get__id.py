import lmdb
import torch
from random import randint
from torch.utils.data import Dataset
import json
import argparse
import  os
import torch.distributed as dist
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
class TextImageDataset(Dataset):
    def __init__(self,
                 all_ids_json='',
                 img_lmdb_name='',
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.img_lmdb_envs=[lmdb.open(img_lmdb_name+'_rank{}'.format(i),
                    max_readers=4096,readonly=True,lock=False,meminit=False,readahead=False) for i in range(72)]
        with open(all_ids_json,'r',encoding='utf-8')as f:
            all_ids=json.load(f)
        self.keys=all_ids
    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        id=self.keys[ind]
        id_=id.encode()
        for i in range(72):
            env = self.img_lmdb_envs[i]
            with env.begin(write=False) as txn:
                img_feat = txn.get(id_)
            if img_feat is None:
                 continue
            return id, str(i)
        return self.skip_sample(ind)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_lmdb_name', type=str, help='model name')
    parser.add_argument('--all_ids_json', type=str, help='model name')

    parser.add_argument('--batch_size', type=int, default=640)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist_url', type=str, help='distributed backend init_method')
    parser.add_argument('--num_workers', type=int, default=8, help='num of wokers for dataloader')
    parser.add_argument('--checkpoint_dir', type=str, default='./ids_json_files/')

    args = parser.parse_args()
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

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
    torch.cuda.set_device(args.local_rank)
    train_dataset = TextImageDataset(
        all_ids_json=args.all_ids_json,
        img_lmdb_name=args.img_lmdb_name,
    )
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
    outfile=os.path.join(args.checkpoint_dir,str(rank)+'.txt')
    idranks=[]
    for i,(ids, lmdb_ranks) in enumerate(train_loader):
        for id,db_rank in zip(ids,lmdb_ranks):
            r=id+'\t'+db_rank
            idranks.append(r)
        if i %200==0:
            with open(outfile,'a',encoding='utf-8')as f:
                for item in idranks:
                    f.write(item)
            idranks=[]
    if idranks !=[]:
        with open(outfile, 'a', encoding='utf-8')as f:
            for item in idranks:
                f.write(item)
        idranks = []



