def compute_suffix_array(input_text, len_text):
    suff = [(i, input_text[i:]) for i in range(len_text)]
    suff.sort(key=lambda x: x[1])
    suffix_arr = [i for i, _ in suff]
    return suffix_arr


def find_last_char(input_text, suffix_arr, n):
    bwt_arr = ""
    for i in range(n):
        j = suffix_arr[i] - 1
        if j < 0:
            j = j + n
        bwt_arr += input_text[j]

    return bwt_arr


def bwt_encode(data):
    data_with_marker = data + '$'
    suffix_arr = compute_suffix_array(data_with_marker, len(data_with_marker))
    bwt_arr = find_last_char(data_with_marker, suffix_arr, len(data_with_marker))
    return bwt_arr


def bwt_decode(bwt_str):
    bwt_list = [(char, i) for i, char in enumerate(bwt_str)]
    bwt_list.sort()
    index_of_end_marker = bwt_str.index('$')
    original_index = index_of_end_marker
    original_str = ""
    for _ in range(len(bwt_str)):
        original_str = bwt_str[original_index] + original_str
        original_index = bwt_list[original_index][1]

    return original_str