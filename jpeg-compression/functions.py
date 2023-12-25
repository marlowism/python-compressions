import cv2
import numpy as np
import numba


def rgb_to_ycbcr(rgb_image):  # step 1
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
def dct(image):  # step 4

    height, width = image.shape
    block_size = 8
    blocks = []

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y + block_size, x:x + block_size]
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
    quantized_block = np.round(block / (quantization_table * a)).astype(np.float32)
    return quantized_block


def quantize_Y(Y, quantization_table_Y, a):  # step 6
    block_size = 8
    height, width = Y.shape
    Y_blocks = []

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = Y[i:i + block_size, j:j + block_size]
            block_flattened = block.flatten()
            Y_blocks.append(block_flattened)

    quantized_Y_blocks = [quantize(block, quantization_table_Y, a) for block in Y_blocks]
    return quantized_Y_blocks


def inverse_zigzag(zigzag_array):
    size = int(np.sqrt(len(zigzag_array)))
    block = np.zeros((size, size), dtype=zigzag_array.dtype)
    index = 0
    for i in range(size):
        if i % 2 == 0:
            for j in range(i + 1):
                block[j, i - j] = zigzag_array[index]
                index += 1
        else:
            for j in range(i + 1):
                block[i - j, j] = zigzag_array[index]
                index += 1
    return block


def inverse_quantize(block, quantization_table, a):
    return block * (quantization_table * a)


@numba.jit(nopython=True)
def inverse_dct(dct_block):
    size = dct_block.shape[0]
    block = np.zeros((size, size), dtype=np.float32)
    for u in range(size):
        for v in range(size):
            sum_idct = 0.0
            for x in range(size):
                for y in range(size):
                    alpha_u = 1.0 if u == 0 else np.sqrt(2) / 2.0
                    alpha_v = 1.0 if v == 0 else np.sqrt(2) / 2.0
                    sum_idct += alpha_u * alpha_v * dct_block[x, y] * np.cos(
                        (2 * x + 1) * u * np.pi / (2 * size)) * np.cos((2 * y + 1) * v * np.pi / (2 * size))
            block[u, v] = sum_idct
    return block


def ycbcr_to_rgb(ycbcr_image):
    Y, Cb, Cr = cv2.split(ycbcr_image)

    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)

    B = Y + 1.772 * (Cb - 128)

    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    return cv2.merge((B, G, R))
