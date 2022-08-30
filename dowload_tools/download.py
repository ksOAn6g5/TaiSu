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

def dowload_pic(pic_url,root):#下载单个图片
	global count
	try:
		suffix ='.jpg' #图片后缀
		name = pic_url[8:48]
		#for i in ['|','<','>','\\','/',':','?','*','"','.']:
		for i in ['/','=','.',',']:
		    name = name.replace(i,'')#去除不能出现在文件名中的字符
		path = root + name+ suffix
		print(path)
		if not os.path.exists(path):#确认是否为新图片
			r = requests.get(pic_url,headers = headers,timeout = (5,50))
			r.raise_for_status()
			with open(path,'wb') as f:
				f.write(r.content)
			print(f'{path}下载成功！',flush = True)
			count+=1
	except:
		print(f'{pic_url}下载失败！',flush = True)


 

def download_pics(pic_urls,root):#采用多线程下载图片
	print('\n开始下载')
	print(f'{len(pic_urls)}个')
	if(len(pic_urls)==0):
		return ''
	#root = 'G:/baidu_data/'#f'/storage-root/platform/baidu_download/baidu//{keyword}//'#默认保存在D盘，可根据需要自行修改
	if not os.path.exists(root):
		os.mkdir(root)#创建存放图片的文件夹
	with ThreadPoolExecutor(max_workers = 50) as pool:#采用50个线程
		for pic_url in pic_urls:
			pool.submit(dowload_pic,pic_url,root)

 



if __name__=='__main__':
    csv_file=urls_csvfile
    root='./images' #
    df = pd.read_csv(csv_file,encoding='utf-8')
    urls=df['url']
    download_pics(urls,root)
    print(count)
           
    #download_keywords(keywords)
