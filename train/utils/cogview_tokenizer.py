# Sentencepiece Chinese Tokenizer
class SentencepieceChineseTokenizer:
    def __init__(self, pretrained_file = "./chinese_sentencepiece/cog-pretrain.model"):
        self.tokenizer = from_pretrained(file=pretrained_file)
        self.vocab_size = self.tokenizer.num_tokens

    def encode(self, text):
        return torch.tensor(self.tokenizer.encode(text))

    def decode(self, tokens):
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()

        tokens = [token for token in tokens if token not in (0,)]
        return self.tokenizer.decode(tokens)

    def tokenize(self, texts, context_length = 256, truncate_text = False):
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = [self.encode(text) for text in texts]

        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate_text:
                    tokens = tokens[:context_length]
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

