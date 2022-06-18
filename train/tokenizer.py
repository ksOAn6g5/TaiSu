# modified from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py


import torch
from transformers import BertTokenizer

# OpenAI simple tokenizer

class ChineseTokenizer:
    def __init__(self):
        tokenizer = torch.load('bert_chinese_tokenizer.pth')#BertTokenizer.from_pretrained('bert-base-chinese')
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

    def decode(self, tokens, pad_tokens = set()):
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
            
        ignore_ids = pad_tokens.union({0})
        tokens = [token for token in tokens if token not in ignore_ids]
        return self.tokenizer.decode(tokens)

    def encode(self, text):
        code=self.tokenizer.encode(text, add_special_tokens = False)
        code.append(21129)
        return torch.tensor(code)#(self.tokenizer.encode(text, add_special_tokens = False))

    def tokenize(self, texts, context_length = 52, truncate_text = True):
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = [self.encode(text) for text in texts]

        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length-1:
                #if truncate_text:
                tokens = tokens[:context_length-1]
                tokens[-1]=21129
               # else:
                #    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, 1:len(tokens)+1] = torch.tensor(tokens)
        result[:,0]=21128
        return result

