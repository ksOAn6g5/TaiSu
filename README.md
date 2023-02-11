# TaiSu(太素--亿级大规模中文视觉语言预训练数据集)
**TaiSu: A 166M Large-scale High-Quality Dataset for Chinese Vision-Language Pre-training**

This paper has been accepted by NeurIPS 2022. 
* paper link: https://openreview.net/pdf?id=iAxH-ikIP0I

* Dataset Construction:
1) Data collection
2) Text-based filtering
3) Image-text-retrieval-based filtering
4) Image-Captioning-based text augmentation
![word cloud](/imgs/all_wc.png)

## Dataset download ##
Since most of the original urls are expired, we decided to directly provide the images and corresponding captions. To make the download process easier, we split the image set into more than 30 parts, and the captions are gathered in a single JSON file whose format of the content is shown in ![captions](/imgs/image.png).  
```
  {id:{'w':*,'g':*}}
  #where 'w' denotes web caption and 'g' denotes generated caption. 
```
All the data can be downloaded by the following link: <https://pan.baidu.com/s/1F5aKsurZkjZie09GsseOlw?pwd=vstf>

The files with the suffix of '.tgz' need first to be uncompressed to a file with the suffix of '.tar' using the command line ```pigz -d baidu_images*.tgz ```.
Even though a part of the images is damaged or lost because of some reasons, you can still access the most part of TaiSu's data. Each image and its captions can be matched by the id, for example, 'img1baiducomitu1848496827104259151'.


## Pretrained models ##
 Models trained on the web data of TaiSu and on the complete data of TaiSu are now availbale.
 Baidu cloud link：https://pan.baidu.com/s/1d3UKyQi7J4Qr1XE2j2V8og?pwd=0kjm 
 * Example for usage:
 ```
 from models.model_infer import build_lit
 from clip.clip import _transform
 from utils.sp_tokenizer import SentencepieceChineseTokenizer
 from PIL import Image
 lit=build_lit(visual_model_path=path/to/visual/model/state_dict,txt_model_path=path/to/textual/model/state_dict)
 #viusal model and textual model should be matched.
 '''API:
    lit.encode_image(imgs)
    lit.encode_text(txt) '''
 device = "cpu"
 transform=_transform(n_px=224)
 tokenizer=SentencepieceChineseTokenizer(context_length=52)
 image = transform(Image.open("xxx.png")).unsqueeze(0).to(device)
 texts = tokenizer.tokenize(['我爱我的家乡','xxxx']).to(device)
 with torch.no_grad():
      img_emb= lit.encode_image(image)
      txt_emb=lit.encode_text(texts)
      #The embeddings should be normalized to calculate cosine similarity
      img_emb=img_emb/img_emb.norm(dim=-1,keepdim=True)
      txt_emb=txt_emb/txt_emb.norm(dim=-1,keepdim=True)
      logits=img_emb@txt_emb.t()     
    
 ```

 
## LICENCE ##
Unless specifically labeled otherwise, these Datasets are provided to You under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (“CC BY-NC-SA 4.0”), with the additional terms included herein. The CC BY-NC-SA 4.0 may be accessed at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode. When You download or use the Datasets from the Website or elsewhere, You are agreeing to comply with the terms of CC BY-NC-SA 4.0, and also agreeing to the Dataset Terms. Where these Dataset Terms conflict with the terms of CC BY-NC-SA 4.0, these Dataset Terms shall prevail. We reiterate once again that this dataset is used only for non-commercial purposes such as academic research, teaching, or scientific publications. We prohibits You from using the dataset or any derivative works for commercial purposes, such as selling data or using it for commercial gain.

`If any of the images belongs to you and you would like it removed, please kindly inform us, we will remove it from our dataset immediately.`

## Contact
  Email:datasets_2022@outlook.com
  
  Organization: Institute of Automation, Chinese Academy of Sciences (CASIA), Beijing, China
## Citation 
```
@inproceedings{
liu2022taisu,
title={TaiSu: A 166M Large-scale High-Quality Dataset for Chinese Vision-Language Pre-training},
author={Yulong Liu and Guibo Zhu and Bin Zhu and Qi Song and Guojing Ge and Haoran Chen and GuanHui Qiao and Ru Peng and Lingxiang Wu and Jinqiao Wang},
booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2022},
url={https://openreview.net/forum?id=iAxH-ikIP0I}
}
```
