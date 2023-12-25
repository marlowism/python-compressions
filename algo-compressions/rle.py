def rle_encode(data):
    encoded_string = ""
    i = 0
    while (i <= len(data)-1):
        count = 1
        ch = data[i]
        j = i
        while (j < len(data)-1):
            if (data[j] == data[j + 1]):
                count = count + 1
                j = j + 1
            else:
                break
        encoded_string = encoded_string + str(count) + ch
        i = j + 1
    return encoded_string

def rle_decode(encoded_data):
    decoded_message = ""
    i = 0
    j = 0
    while (i <= len(encoded_data) - 1):
        run_count = int(encoded_data[i])
        run_word = encoded_data[i + 1]
        for j in range(run_count):
            decoded_message = decoded_message+run_word
            j = j + 1
        i = i + 2
    return decoded_message

