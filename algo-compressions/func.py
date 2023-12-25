from ac import*
from huffman import*
from lz78 import*
from mtf import *
from bwt import*
from rle import*


def huffman(data):
    encoded_data = HuffmanEncoding(data)
    return len(encoded_data[0])

def AC_helper(data):
    block_size = 100
    encoder = ArithmeticEncoder(data, block_size)
    encoded_data = encoder.encode()
    encoder.save_probability_table("probability_table.txt")
    return len(encoded_data*8)

def lz78(data):
    encode=lz78_encode(data)
    return print("lz78:",(len(encode)*8),"bit")

def mtf(data):
    encode = mtf_encode(data)
    return encode

def bwt(data):
    encode = bwt_encode(data)
    return encode

def rle(data):
    return rle_encode(data)

def bwt_mtf_ha(data):
    encoded_bwt=bwt(data)
    encoded_mtf=mtf(encoded_bwt)
    string = ''.join(str(item) for item in encoded_mtf)
    result=huffman(string)
    print("bwt_mtf_ha:",result,"bit")

def bwt_mtf_ac(data):
    encoded_bwt = bwt(data)
    encoded_mtf = mtf(encoded_bwt)
    string = ''.join(str(item) for item in encoded_mtf)
    result=AC_helper(string)
    print("bwt_mtf_ac:",result,"bit")

def rle_bwt_mtf_rle_ha(data):
    encoded_rle = rle_encode(data)
    encoded_bwt=bwt(encoded_rle)
    encoded_mtf=mtf(encoded_bwt)
    string = ''.join(str(item) for item in encoded_mtf)
    encoded_rle=rle(string)
    encoded_ha=huffman(encoded_rle)
    print("rle_bwt_mtf_rle_ha",encoded_ha,"bit")

def rle_bwt_mtf_rle_ac(data):
    encoded_rle=rle(data)
    encoded_bwt=bwt(encoded_rle)
    encoded_mtf=mtf(encoded_bwt)
    string = ''.join(str(item) for item in encoded_mtf)
    encoded_rle=rle(string)
    encoded_ac=AC_helper(encoded_rle)
    print("rle_bwt_mtf_rle_ac",encoded_ac,"bit")