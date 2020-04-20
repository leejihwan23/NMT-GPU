
import sentencepiece as spm
from .tokenizer import Korean_tokenizer, English_tokenizer

def Korean_tokenizer_load(x):

    sp = spm.SentencePieceProcessor()
    sp.Load('/home/team012/LJH/NMT4-20/data/korean_tok.model')

    return sp.EncodeAsPieces(x)


def English_tokenizer_load(x):
    
    sp = spm.SentencePieceProcessor()
    sp.Load('/home/team012/LJH/NMT4-20/data/english_tok.model')

    return sp.EncodeAsPieces(x)
