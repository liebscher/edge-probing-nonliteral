import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')

import numpy as np
import torch as tt
import transformers as tm

def embedder_from_str(name):
    if name == "BERT":
        return BERTEmbedder()
    elif name == "ELMo":
        return ELMoEmbedder()
    else:
        raise TypeError(f"Given embedder name does not exist: \"{name}\"")

class Embedder():
    def __init__(self, tokenizer, transformer):
        self._tokenizer = tokenizer
        self._transformer = transformer

    def tokenize(self, sequence):
        return tt.tensor([self._tokenizer.encode(sequence)])

    def embed(self, sequence, layer=0):
        with tt.no_grad():
            return self._transformer(sequence)[layer]

    def get_dims(self):
        pass


class BERTEmbedder(Embedder):
    def __init__(self):
        super(BERTEmbedder, self).__init__(
            tm.BertTokenizer.from_pretrained("bert-base-uncased"),
            tm.BertModel.from_pretrained("bert-base-uncased"))

        logger.debug("Initializing new BERT Embedder")

    def get_dims(self):
        return 0

class ELMoEmbedder(Embedder):
    def __init__(self):
        super(ELMoEmbedder, self).__init__(
            tm.BertTokenizer.from_pretrained("bert-base-uncased"),
            tm.BertModel.from_pretrained("bert-base-uncased"))

        logger.debug("Initializing new ELMo Embedder")

    def get_dims(self):
        return 0
