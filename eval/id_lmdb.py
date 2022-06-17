# -*- coding:utf-8 -*-
import lmdb
import  json
import numpy as np
class lmdb_handle():
    def __init__(self,db_name):
        self.env = lmdb.open(db_name)
        if not self.env:
            raise IOError('Cannot open lmdb dataset', db_name)
    def add_id_rank(self,cache):
       # c=0
        with self.env.begin(write=True) as txn:
            for (k, v) in cache:
               #print('k:',k,'v',v)
                txn.put(k, v)
    def close(self):
        self.env.close()
    def set_mapsize(self,size):
        self.env.set_mapsize(size)

id_rank_json=''
with open(id_rank_json,'r',encoding='utf-8')as f:
    id2rank=json.load(f)
#id2rank={'sjfkja':2,'adh':3}
cache=[]
db=lmdb_handle('id2rank_lmdb')
for i,(img_id,rank) in enumerate(id2rank.items()):
    cache.append((str(i).encode(),np.array([img_id,str(rank)])))
    if i%1==0:
        db.add_id_rank(cache)
        cache=[]

