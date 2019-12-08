import torch as tt
import transformers as tm

import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')


def embedder_from_str(name):
    """

    Parameters
    ----------
    name

    Returns
    -------

    """
    if name == "BERT":
        return BERTEmbedder()
    elif name == "BERTL":
        return BERTLargeEmbedder()
    elif name == "GPT2":
        return GPT2Embedder()
    elif name == "ELMo":
        return ELMoEmbedder()
    else:
        raise TypeError(f"Given embedder name does not exist: \"{name}\"")


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class Embedder():
    def __init__(self, tokenizer, transformer):
        """

        Parameters
        ----------
        tokenizer : transformers
        transformer : transformers
        """
        self._tokenizer = tokenizer
        self._transformer = transformer

    def tokenize(self, sequence):
        """
        Tokenize a string, first by whitespaces and then by the tokenizer's built-in tokenizer

        Parameters
        ----------
        sequence : string
            The input sequence to tokenize

        Returns
        -------
        ind : list
            A map between whitespaced tokens and a further tokenized sequence
        tokenized :
            The sequence still in human-readable format, but tokenized
        encoded : torch.tensor (1, k, d)
            The tokenized sequence in BOW format
        """
        ind = []  # the indice map of length of encoding, each element represent which original token
        encoded_seq = []  # encoded sequence

        # we iterate overall all tokens split by simple whitespace
        for i, tok in enumerate(whitespace_tokenize(sequence)):
            # using the encoder, get vocabulary IDs
            enc = self._tokenizer.encode(tok)
            # build up map, original tokens get 1+ encoded tokens (may be split up)
            ind.extend([i] * len(enc))

            encoded_seq.extend(enc)

        return ind, self._tokenizer.tokenize(sequence), tt.tensor([encoded_seq])

    def detokenize(self, tokens):
        """

        Parameters
        ----------
        tokens :

        Returns
        -------

        """
        return self._tokenizer.decode(tokens[0].tolist())

    def embed(self, sequence, layer=0):
        """
        Embed a sequence of encoded tokens in a latent space, defined by a transformer

        Parameters
        ----------
        sequence : torch.tensor (1, k, d)
        layer : int

        Returns
        -------

        """
        with tt.no_grad():
            return self._transformer(sequence)[layer]

    def get_dims(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class BERTEmbedder(Embedder):
    def __init__(self):
        """
        Use a BERT embedder from huggingface
        """
        super(BERTEmbedder, self).__init__(
            tm.BertTokenizer.from_pretrained("bert-base-uncased", do_basic_tokenize=True),
            tm.BertModel.from_pretrained("bert-base-uncased"))

        logger.debug("Initializing new BERT Embedder")

    def get_dims(self):
        return 768

    def __str__(self):
        return "BERT"


class BERTLargeEmbedder(Embedder):
    def __init__(self):
        """
        Use a BERT embedder from huggingface
        """
        super(BERTLargeEmbedder, self).__init__(
            tm.BertTokenizer.from_pretrained("bert-large-uncased", do_basic_tokenize=True),
            tm.BertModel.from_pretrained("bert-large-uncased"))

        logger.debug("Initializing new BERT Large Embedder")

    def get_dims(self):
        return 1024

    def __str__(self):
        return "BERT-Large"


class GPT2Embedder(Embedder):
    def __init__(self):
        """
        Use a GPT2 embedder from huggingface
        """
        super(GPT2Embedder, self).__init__(
            tm.GPT2Tokenizer.from_pretrained("gpt2", do_basic_tokenize=True),
            tm.GPT2Model.from_pretrained("gpt2"))

        logger.debug("Initializing new GPT2 Embedder")

    def get_dims(self):
        return 768

    def __str__(self):
        return "GPT2"


class ELMoEmbedder(Embedder):
    def __init__(self):
        """
        Use an ELMo embedder
        TODO
        """
        # elmo = ElmoEmbedder()

        super(ELMoEmbedder, self).__init__(
            tm.BertTokenizer.from_pretrained("bert-base-uncased"),
            tm.BertModel.from_pretrained("bert-base-uncased")
        )

        logger.debug("Initializing new ELMo Embedder")

    def get_dims(self):
        return 0
