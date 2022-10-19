import requests
import re
import os
from concurrent.futures import ThreadPoolExecutor#线程池
import time
import json
import pickle
import pandas as pd

#headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36  Edg/89.0.774.77'}#请求头
headers = {
    #'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    'User-Agent':'Googlebot-Image/1.0', # Pretend to be googlebot
    'X-Forwarded-For': '64.18.15.200'
}

count=0

def dowload_pic(pic_url,root):#download a single pic
	global count
	try:
		suffix ='.jpg' #图片后缀
		name = pic_url[8:48]
		#for i in ['|','<','>','\\','/',':','?','*','"','.']:
		for i in ['/','=','.',',']:
		    name = name.replace(i,'')#delete illegal characters
		path = root + name+ suffix
		print(path)
		if not os.path.exists(path):
			r = requests.get(pic_url,headers = headers,timeout = (5,50))
			r.raise_for_status()
			with open(path,'wb') as f:
				f.write(r.content)
			print(f'{path}sucessfully downloaded！',flush = True)
			count+=1
	except:
		print(f'{pic_url}failure！',flush = True)


 

def download_pics(pic_urls,root):#multi-threads
	print('\n begin')
	print(f'{len(pic_urls)}urls')
	if(len(pic_urls)==0):
		return ''
	if not os.path.exists(root):
		os.mkdir(root)#creat directory
	with ThreadPoolExecutor(max_workers = 10) as pool:
		for pic_url in pic_urls:
			pool.submit(dowload_pic,pic_url,root)

 



if __name__=='__main__':
    csv_file=urls_csvfile
    root='./images' #
    df = pd.read_csv(csv_file,encoding='utf-8')
    urls=df['url']
    download_pics(urls,root)
    print(count)
           
