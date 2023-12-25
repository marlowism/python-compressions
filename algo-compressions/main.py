import plot
from func import*
import time
from plot import*

def compress_time(function, *args, **kwargs):
    start_time = time.time()
    result = function(*args, **kwargs)
    end_time = time.time()
    res_time = end_time - start_time
    print(f"time: {res_time*1000} m-seconds\n")
    return result


with open('enwik8.txt', 'r', encoding='utf-8') as file:
    enwik8 = "".join(next(file) for _ in range(100))



print(f"enwik8 size: {len(enwik8)*8} bit\n")

print("huffman:",huffman(enwik8),"bit")
huffman_time = compress_time(huffman, enwik8)

print("ac:",AC_helper(enwik8),"bit")
ac_time = compress_time(AC_helper, enwik8)

lz78_time = compress_time(lz78, enwik8)

rle_bwt_mtf_rle_ha_time = compress_time(rle_bwt_mtf_rle_ha, enwik8)

bwt_mtf_ha_time = compress_time(bwt_mtf_ha, enwik8)

bwt_mtf_ac_time = compress_time(bwt_mtf_ac, enwik8)

rle_bwt_mtf_rle_ac_time = compress_time(rle_bwt_mtf_rle_ac, enwik8)

plot.plots()

