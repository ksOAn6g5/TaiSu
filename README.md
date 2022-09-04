# TaiSu
TaiSu（太素）--A 166M multimodal dataset for Chinese Vision-Language Pretraining
* Dataset Construction:
1) Data collection
2) Text-based filtering
3) Image-text-retrieval-based filtering
4) Image-Captioning-based text augmentation
![word cloud](/imgs/all_wc.png)

## Dataset download ##
Taisu data is available now.The image urls and corresponding texts are stored in a CSV file. 
Baidu cloud link:
* URLs&captions for TaiSu dataset: https://pan.baidu.com/s/1YITGlMF2L7EFLZrLuETJKQ?pwd=tais

## Pre-extracted image embeddings
We provide the image embeddings extracted with CLIP's RN101 and ViT-B/32 variants. 
* Pre-extracted image features: 
## Pretrained models ##
 Models trained on the web data of TaiSu and on the complete data of TaiSu are now availbale.
 Baidu cloud link：https://pan.baidu.com/s/1d3UKyQi7J4Qr1XE2j2V8og?pwd=0kjm 
 * for utilization:
 ```
 from models.model_infer import build_lit
 lit=build_lit(visual_model_path=path/to/visual/model/state_dict,txt_model_path==path/to/textual/model/state_dict)
 API:
    lit.encode_image(imgs)
    lit.encode_text(txt) 
 ```
 
## LICENCE ##
Unless specifically labeled otherwise, these Datasets are provided to You under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (“CC BY-NC-SA 4.0”), with the additional terms included herein. The CC BY-NC-SA 4.0 may be accessed at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode. When You download or use the Datasets from the Website or elsewhere, You are agreeing to comply with the terms of CC BY-NC-SA 4.0, and also agreeing to the Dataset Terms. Where these Dataset Terms conflict with the terms of CC BY-NC-SA 4.0, these Dataset Terms shall prevail. We reiterate once again that this dataset is used only for non-commercial purposes such as academic research, teaching, or scientific publications. We prohibits You from using the dataset or any derivative works for commercial purposes, such as selling data or using it for commercial gain.
## Contact
  Email:datasets_2022@outlook.com
  Organization: Institute of Automation, Chinese Academy of Sciences (CASIA), Beijing, China
## Citation 
