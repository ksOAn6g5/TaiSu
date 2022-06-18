import lmdb
import torch
from random import randint
from torch.utils.data import Dataset
import numpy as np
#import json
class TextImageDataset(Dataset):
    def __init__(self,
                 split='only_web_txt',
                 id2rank_lmdb='',
                 img_lmdb_name='',
                 text_lmdb_folder='',
                 tokenizer=None,
                 shuffle=False
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.split=split
        self.tokenizer=tokenizer
        self.shuffle = shuffle
        self.img_lmdb_envs=[lmdb.open(img_lmdb_name+'_rank{}'.format(i),
                    max_readers=4096,readonly=True,lock=False,meminit=False,readahead=False) for i in range(48)]
        self.text_lmdb_env= lmdb.open(text_lmdb_folder,max_readers=4096,readonly=True,lock=False,meminit=False,readahead=False)
        self.id2rank_env=lmdb.open(id2rank_lmdb,max_readers=4096,readonly=True,lock=False,meminit=False,readahead=False)


        # print([int(176625970),int(110730104)][self.split])
        # print([int(166587549),int(133071474)][self.split])



        # if self.split=='None':
        #     print(170907761) 
        # else:
        #     print(105011895)
            
    def __len__(self):
         return [int(165150664),int(133071474)][self.split] 
        # return [int(176625970),int(110730104)][self.split]
        # return [int(170907761),int(105011895)][self.split]
        # else:
        #     return int(105011895)
        # if split == 'only_web_txt':
        #     return int(105011895)
        # if split == 'None' :
        #     return int(170907761)
        # return int(105011895)      
        #return len(self.img_ids)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >(self.__len__() - 2):
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        index=str(ind).encode()
        with self.id2rank_env.begin(write=False) as txn:
            idrank=txn.get(index)
        if idrank==None: 
            print('idrank is None,skipping {}'.format(ind))
            return self.skip_sample(ind)
        type = '<U{}'.format(len(idrank) // 8)
        idrank=np.frombuffer(idrank,dtype=type)
        key=idrank[0]
        rank=int(idrank[1])
        #key= self.img_ids[ind]
        #lmdbind=int(self.ranks[ind])
        #print('lmdb index:',lmdbind)
        env =self.img_lmdb_envs[rank]
        #print(env)
        key = key.encode()

        try:
            with env.begin(write=False) as txn:
                img_feat = txn.get(key)
            if img_feat is None:
                print('img_feat is None,skipping {}'.format(key))
                return self.skip_sample(ind)
            img_feat = np.frombuffer(img_feat, dtype=np.float32)
            img_feat = torch.from_numpy(img_feat)
            with self.text_lmdb_env.begin(write=False) as txn:
                txt = txn.get(key)
            if txt == None:
                print('txt is None,skipping {}'.format(key))
                return self.skip_sample(ind)
            txt=txt.decode()
            if '#$#' in txt:# web text and generated text are seperated with '#$#'
                txt=txt.split('#$#')
                if self.split==1:
                   txt=txt[-1]
                else:
                   txt=txt[randint(0,1)]
            txt = self.tokenizer.tokenize(txt)[0]
            return txt, img_feat
        except:
            print('error,skipping {}'.format(key))
            return self.skip_sample(ind)
       


if __name__=='__main__':
    from torch.utils.data import DataLoader
    #from utils.sp_tokenizer import SentencepieceChineseTokenizer
    from tokenizer import ChineseTokenizer
    tokenizer=ChineseTokenizer()
    #id2img_lmdb_json='/public/home/actqrzwa6p/CLIP_zb/AAA_LiT/yulong/CLIP_embed/baidu_lmdb_rank_0.json'# ###save_len2_50_in_rank_all2.json 
    img_lmdb_name='/work1/multimodel/baidu_images_features/baidu_lmdb'#填32个文件夹的公共路径+_rank之前的名字部分
    text_lmdb_folder='/public/home/multimodel/multimodel2/web_ofa_v2_str'
    id2rank_lmdb='/public/home/multimodel/multimodel2/clean_web+clean_ofa_id_rank_RN'
    ds=TextImageDataset(0,id2rank_lmdb=id2rank_lmdb,
                 img_lmdb_name=img_lmdb_name,
                 text_lmdb_folder=text_lmdb_folder,
                 tokenizer=tokenizer,
                 shuffle=False)
    print(len(ds))
    loader=DataLoader(ds,batch_size=1)
    print(len(loader))
    for i, data in enumerate(loader):
       txt,img = data
       print('txt:',txt.shape,'img.shape:',img.shape)
       #if i==10:break


#'''生成数据库'''
        #print(att)
    # id2lmdb={}
    # for i in range(32):
    #     name='./baidu/baidu_lmdb_rank'+str(i)
    #     env=lmdb.open(name)
    #     with env.begin(write=True) as txn:
    #         v=np.zeros(512,dtype=np.float32)+np.random.random()
    #         txn.put(str(i).encode(),v)
    #     id2lmdb[str(i)]=i
    #     env.close()
    # env=lmdb.open('id2caption')
    # with env.begin(write=True) as txn:
    #        for i in range(32):
    #            txn.put(str(i).encode(),np.array(['这不是开玩笑'+str(np.random.random()), '这是咋爱开玩笑'+str(np.random.random())]))
    # env.close()
    # with open('id2lmdb','w',encoding='utf-8')as f:
    #     json.dump(id2lmdb,f)
