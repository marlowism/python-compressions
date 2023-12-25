import cv2
import numpy as np
import numba

def rgb_to_ycbcr(rgb_image):   # step 1
    B, G, R = cv2.split(rgb_image)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    Y = np.round(Y).astype(np.uint8)
    Cb = np.round(Cb).astype(np.uint8)
    Cr = np.round(Cr).astype(np.uint8)

    ycbcr_image = cv2.merge((Y, Cb, Cr))

    return ycbcr_image, Y

def subsample(ycbcr_image):  # step 2

    Y, Cb, Cr = cv2.split(ycbcr_image)

    height, width = Y.shape

    Cb_subsampled = np.zeros((height // 2, width // 2), dtype=np.uint8)
    Cr_subsampled = np.zeros((height // 2, width // 2), dtype=np.uint8)

    for i in range(0, height, 2):
        for j in range(0, width, 2):
            Cb_subsampled[i // 2, j // 2] = np.mean(Cb[i:i + 2, j:j + 2])
            Cr_subsampled[i // 2, j // 2] = np.mean(Cr[i:i + 2, j:j + 2])

    return Cb_subsampled, Cr_subsampled

def to_signed(subsampled_value):  # step 3

    signed_value = subsampled_value - 128
    signed_value = np.clip(signed_value, -128, 127)
    signed_value = signed_value.astype(np.int8)

    return signed_value

@numba.jit(nopython=True)
def dct(image):         # step 4

    height, width = image.shape
    block_size = 8
    blocks = []

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            blocks.append(block)

    dct_blocks = []

    for block in blocks:
        dct_result = np.zeros((block_size, block_size), dtype=np.float32)

        for u in range(block_size):
            for v in range(block_size):
                sum_dct = 0.0
                for x in range(block_size):
                    for y in range(block_size):
                        sum_dct += block[x, y] * np.cos((2 * x + 1) * u * np.pi / (2 * block_size)) * \
                                   np.cos((2 * y + 1) * v * np.pi / (2 * block_size))

                alpha_u = 1.0 if u == 0 else np.sqrt(2) / 2.0
                alpha_v = 1.0 if v == 0 else np.sqrt(2) / 2.0

                sum_dct *= (alpha_u * alpha_v * 2.0 / block_size)

                dct_result[u, v] = sum_dct

        dct_blocks.append(dct_result)

    return dct_blocks

def zigzag(block):  # step 5
    size = block.shape[0]
    result = np.zeros(8 * 8, dtype=block.dtype)
    index = 0
    for i in range(size):
        if i % 2 == 0:
            for j in range(i + 1):
                result[index] = block[j, i - j]
                index += 1
        else:
            for j in range(i + 1):
                result[index] = block[i - j, j]
                index += 1
    return result

def quantize(block, quantization_table, a):  # step 6
    quantized_block = np.round(block / (quantization_table * a)).astype(np.int32)
    return quantized_block

def quantize_Y(Y, quantization_table_Y, a):  # step 6
    block_size = 8
    height, width = Y.shape
    Y_blocks = []

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = Y[i:i+block_size, j:j+block_size]
            block_flattened = block.flatten()
            Y_blocks.append(block_flattened)

    quantized_Y_blocks = [quantize(block, quantization_table_Y, a) for block in Y_blocks]
    return quantized_Y_blocks

def RLE(block):   # step 7

    encoded_data = []
    zero_count = 0

    for value in block:
        if value == 0:
            zero_count += 1
        else:

            while zero_count > 15:
                encoded_data.append((15, 0))
                zero_count -= 16
            encoded_data.append((zero_count, value))
            zero_count = 0

    encoded_data.append((0, 0))
    return encoded_data

def decode_dc_RLE(encoded_data):
    decoded_dc = []

    for run_length, value in encoded_data:
        if run_length == 0:
            decoded_dc.append(value)
        else:
            zeros = [0] * run_length
            decoded_dc.extend(zeros)
            decoded_dc.append(value)

    return decoded_dc

def decode_ac_RLE(encoded_data):
    decoded_block = []
    zero_count = 0

    for zero_count, value in encoded_data:
        if zero_count == 0 and value == 0:
            decoded_block.extend([0] * (64 - len(decoded_block)))
            break

        zeros = [0] * zero_count
        decoded_block.extend(zeros + [value])

    return decoded_block


def inverse_quantize(block, quantization_table, a):
    return block * (quantization_table * a)


def inverse_zigzag(blocks, block_size, image_size):
    image = np.zeros(image_size, dtype=blocks[0].dtype)
    block_index = 0
    for row in range(image_size[0] // block_size):
        for col in range(image_size[1] // block_size):
            for i in range(block_size):
                for j in range(block_size):
                    image[row * block_size + i, col * block_size + j] = blocks[block_index][i * block_size + j]
            block_index += 1
    return image

def inverse_zigzag_ac(blocks, block_size, image_size):
    height, width = image_size
    image = np.zeros(image_size, dtype=np.int32)

    block_index = 0
    for row in range(height // block_size):
        for col in range(width // block_size):
            for i in range(block_size):
                for j in range(block_size):
                    if block_index < len(blocks):
                        image[row * block_size + i, col * block_size + j] = blocks[block_index][i * block_size + j]
                    else:
                        break
            block_index += 1

    return image






img = 'input.png'

rgb_image = cv2.imread(img)

ycbcr_image, Y = rgb_to_ycbcr(rgb_image)


cb_subsampled, cr_subsampled = subsample(ycbcr_image)

cb_subsampled = to_signed(cb_subsampled)
cr_subsampled = to_signed(cr_subsampled)

dct_cb_blocks = dct(cb_subsampled)
dct_cr_blocks = dct(cr_subsampled)

zigzag_cb_blocks = [zigzag(block) for block in dct_cb_blocks]
zigzag_cr_blocks = [zigzag(block) for block in dct_cr_blocks]

quantization_table_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32).flatten()

quantization_table_CbCr = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32).flatten()

a = 0.2   # коэф квантования

height, width, _ = rgb_image.shape

quantized_cb_blocks = [quantize(block, quantization_table_CbCr, a) for block in zigzag_cb_blocks]
quantized_cr_blocks = [quantize(block, quantization_table_CbCr, a) for block in zigzag_cr_blocks]
quantized_Y_blocks = quantize_Y(Y, quantization_table_Y, a)


encoded_cb_dc = RLE(quantized_cb_blocks[0])
encoded_cb_ac = [RLE(block) for block in quantized_cb_blocks[1:]]

encoded_cr_dc = RLE(quantized_cr_blocks[0])
encoded_cr_ac = [RLE(block) for block in quantized_cr_blocks[1:]]


dequantized_Y_blocks = [inverse_quantize(block, quantization_table_Y, a) for block in quantized_Y_blocks]


decoded_cb_dc = decode_dc_RLE(encoded_cb_dc)
decoded_cr_dc = decode_dc_RLE(encoded_cr_dc)


decoded_cb_ac = [decode_ac_RLE(block) for block in encoded_cb_ac]
decoded_cr_ac = [decode_ac_RLE(block) for block in encoded_cr_ac]

inverse_quantized_cb_dc = np.array(decoded_cb_dc)
inverse_quantized_cr_dc = np.array(decoded_cr_dc)

inverse_quantized_cb_ac = [np.array(block) for block in decoded_cb_ac]
inverse_quantized_cr_ac = [np.array(block) for block in decoded_cr_ac]

inverse_cb_ac_image = inverse_zigzag_ac(inverse_quantized_cb_ac, 8, (height // 2, width // 2))
inverse_cr_ac_image = inverse_zigzag_ac(inverse_quantized_cr_ac, 8, (height // 2, width // 2))

inverse_cb_dc_image = inverse_quantized_cb_dc
inverse_cr_dc_image = inverse_quantized_cr_dc


upsampled_cb = cv2.resize(inverse_cb_dc_image, (width, height), interpolation=cv2.INTER_NEAREST)

upsampled_cr = cv2.resize(inverse_cr_dc_image, (width, height), interpolation=cv2.INTER_NEAREST)


# Шаг 3: Комбинирование компонент Y, Cb и Cr
reconstructed_ycbcr_image = np.zeros((height, width, 3), dtype=np.uint8)

# Повторяем каждый блок Y для соответствия размеру изображения
for i in range(0, height, 8):
    for j in range(0, width, 8):
        reconstructed_ycbcr_image[i:i+8, j:j+8, 0] = dequantized_Y_blocks[i//8 * (width//8) + j//8].reshape(8, 8)

reconstructed_ycbcr_image[:, :, 1] = upsampled_cb
reconstructed_ycbcr_image[:, :, 2] = upsampled_cr



# Шаг 4: Коррекция типов данных и масштабирование
reconstructed_ycbcr_image = np.clip(reconstructed_ycbcr_image, 0, 255).astype(np.uint8)

# Шаг 5: Сохранение восстановленного изображения
cv2.imwrite('output.jpg', reconstructed_ycbcr_image)


#cv2.imwrite('output.jpg', ycbcr_image)
