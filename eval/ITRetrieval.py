"""
Evaluation code for multimodal-ranking
"""
import numpy
import json
import torch
from utils.sp_tokenizer import SentencepieceChineseTokenizer
import torch.distributed as dist

def encode_sentences(model,sent_list,tokenizer,device='cuda'):
    l=len(sent_list)
    batch_size=500
    if l%500==0:
        fois=l//batch_size
    else:
        fois=l//batch_size+1
    if dist.is_initialized():
        m=model.module
    else:
        m=model
    res=[]
    with torch.no_grad():
       for i in range(fois):
           sens= sent_list[i*batch_size:(i+1)*batch_size]
           #sens=['京东商城 '+ item for item in sens]
           sens=tokenizer.tokenize(sens).to(device)
           with amp.autocast(enabled=True):
               embs=m.encode_text(sens)
           res.append(embs)
    res=torch.cat(res,dim=0).cpu()
    return res
def get_eval_data_info(data_name,viusal_model_name):
    with open('preprocess/data_info.json','r',encoding='utf-8')as f:
        d=json.load(f)
    info=d[data_name]
    if data_name in ['flickr8k','flickr30k','AIC']:
        return info["caps_file"],info["img_feats_file"][viusal_model_name]
    else:
        return info["caps_file"], info["img_feats_file"][viusal_model_name],\
               info["target_img_file"],info["target_txt_file"]



def evalrank(model, tokenizer,data_name,viusal_model_name):
    """
    Evaluate a trained model on either dev or test
    data options: f8k, f30k, coco
    """
    if data_name  in ['flickr8k','flickr30k','AIC']:
        cap_file,img_feats_file,=get_eval_data_info(data_name,viusal_model_name)
    else:
        cap_file, img_feats_file, target_img_file,target_txt_file= get_eval_data_info(data_name,viusal_model_name)
    with open(cap_file,'r',encoding='utf-8')as f:
        sent_list=json.load(f)
    img_emb=torch.load(img_feats_file)
    img_emb=torch.from_numpy(img_emb)
    print('Computing results...')
    txt_emb= encode_sentences(model,sent_list,tokenizer,device='cuda')
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    print(data_name)
    if data_name in ['flickr8k','flickr30k','AIC']:
        (r1, r5, r10, medr) = i2t(img_emb, txt_emb)
        print("Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))
        (r1i, r5i, r10i, medri) = t2i(img_emb, txt_emb)
        print("Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri))
    elif data_name=='muge':
        with open(target_img_file,'r',encoding='utf-8')as f:
            target=json.load(f)
        (r1i, r5i, r10i, medri)=t2i_muge(img_emb,txt_emb,target)
        print("Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri))
    else:
        with open(target_img_file,'r',encoding='utf-8')as f:
            target_img=json.load(f)
        with open(target_txt_file, 'r', encoding='utf-8')as f:
            target_txt = json.load(f)
        (r1, r5, r10, medr) = i2t_coco_cn(img_emb, txt_emb,target_txt)
        print("Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))
        (r1i, r5i, r10i, medri) = t2i_coco_cn(img_emb, txt_emb,target_img)
        print("Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri))
def i2t(images, captions, npts=None):
    """
    Images->Text (Image Annotation)
    Images: (N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    npts = images.shape[0]
    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[index].reshape(1, images.shape[1])

        # Compute scores
        d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]

        # Score
        rank = 1e20
        for i in range(5*index, 5*index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)

def t2i(images, captions, npts=None):
    """
    Text->Images (Image Search)
    Images: (N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    npts = images.shape[0]
    #ims = numpy.stack([images[i] for i in range(0, len(images), 5)])
    #print(ims.shape)
    ranks = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5*index : 5*index + 5]

        # Compute scores
        d = numpy.dot(queries, images.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)

def t2i_muge(images, captions,target_img_index):
    """
    Images->Text (Image Annotation)
    Images: (30000, K) matrix of images
    Captions: (5000, K) matrix of captions
    """
    npts = captions.shape[0]
    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query caption
        cap = captions[index].reshape(1, captions.shape[1])

        # Compute scores
        d = numpy.dot(cap, images.T).flatten()
        inds = numpy.argsort(d)[::-1]#降序排列的index

        # Score
        rank = 1e20
        targets=target_img_index[index]
        for i in targets:
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)
def i2t_coco_cn(images, captions,target_txt_index):
    """
    Images->Text (Image Annotation)
    Images: (30000, K) matrix of images
    Captions: (5000, K) matrix of captions
    """
    npts = images.shape[0]
    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query caption
        img = images[index].reshape(1, images.shape[1])

        # Compute scores
        d = numpy.dot(img, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]#降序排列的index

        # Score
        rank = 1e20
        targets=target_txt_index[index]
        for i in targets:
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)

def t2i_coco_cn(images, captions,target_img_index):
    """
    Images->Text (Image Annotation)
    Images: (30000, K) matrix of images
    Captions: (5000, K) matrix of captions
    """
    npts = captions.shape[0]
    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query caption
        cap = captions[index].reshape(1, captions.shape[1])

        # Compute scores
        d = numpy.dot(cap, images.T).flatten()
        inds = numpy.argsort(d)[::-1]#降序排列的index

        # Score
        rank = 1e20
        targets=target_img_index[index]
        # for i in targets:
        tmp = numpy.where(inds == targets)[0][0]
        if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)

p6="G:\ckpt\\vit_web+ofa_lit119.pt"
p7='G:/ckpt/rn101_wp_lit25.pt'
p8='G:/ckpt/vit_wp_lit25.pt'


if __name__=="__main__":
    from models.model_infer import build_lit
    import torch.cuda.amp as amp
    from bert_tokenizer import ChineseTokenizer
    lit=build_lit(visual_model_path='./models/ViT-B-32.pth',txt_model_path=p8)
    lit=lit.cuda()
    tokenizer=ChineseTokenizer()
    for data_name in ['flickr8k','flickr30k','muge','coco-cn']:
   # for data_name in ['coco-cn']:
        evalrank(lit,tokenizer,data_name,'ViT-B-32')
