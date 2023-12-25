def lz78_encode(data):
    dictionary = {}
    encode = []
    buffer = ""
    code = 0

    for char in data:
        buffer += char
        if buffer not in dictionary:
            dictionary[buffer] = code
            if buffer[:-1] != "":
                encode.append((dictionary[buffer[:-1]], char))
            else:
                encode.append((-1, char))
            buffer = ""
            code += 1

    return encode

def lz78_decode(encode_data):
    dictionary = {0: ""}
    decode = []
    code = 0
    buffer = ""

    for entry in encode_data:
        index, char = entry
        if index == -1:
            phrase = char
        else:
            phrase = dictionary[index] + char

        decode.append(phrase)
        dictionary[code + 1] = phrase
        code += 1

    return ''.join(decode)