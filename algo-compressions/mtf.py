def mtf_encode(data):
    alphabet = list(set(data))
    encoded_data = []

    for char in data:
        char_index = alphabet.index(char)
        encoded_data.append(char_index)

        del alphabet[char_index]
        alphabet.insert(0, char)

    return encoded_data


def mtf_decode(encoded_data):
    alphabet = list(range(256))
    decoded_data = []

    for index in encoded_data:
        char = alphabet[index]
        decoded_data.append(char)

        del alphabet[index]
        alphabet.insert(0, char)

    return "".join(decoded_data)
