import json
from tqdm import tqdm
ids=[]
cnt=0
for i in tqdm(range(200)):
    path='./filtered_ofa/cleaned_ids{}.txt'.format(i)
    with open(path,'r',encoding='utf-8')as f:
         all=f.readlines()
         for j in all:
            ids.append(j.replace('\n',''))
            cnt+=1
print(cnt)
with open('ofa_filtered_ids.json','w',encoding='utf-8')as f:
     json.dump(ids,f)

