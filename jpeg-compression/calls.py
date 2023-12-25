from functions import *
from table import *


def jpeg_comperession(a):
    img = 'input.png'

    rgb_image = cv2.imread(img)

    ycbcr_image, Y = rgb_to_ycbcr(rgb_image)


    cb_subsampled, cr_subsampled = subsample(ycbcr_image)


    cb_subsampled = to_signed(cb_subsampled)
    cr_subsampled = to_signed(cr_subsampled)



    dct_cb_blocks = dct(cb_subsampled)
    dct_cr_blocks = dct(cr_subsampled)


    print(dct_cr_blocks)

    zigzag_cb_blocks = [zigzag(block) for block in dct_cb_blocks]
    zigzag_cr_blocks = [zigzag(block) for block in dct_cr_blocks]


    height, width, _ = rgb_image.shape

    quantized_cb_blocks = [quantize(block, quantization_table_CbCr, a) for block in zigzag_cb_blocks]
    quantized_cr_blocks = [quantize(block, quantization_table_CbCr, a) for block in zigzag_cr_blocks]
    quantized_Y_blocks = quantize_Y(Y, quantization_table_Y, a)



    dequantized_Y_blocks = [inverse_quantize(block, quantization_table_Y, a) for block in quantized_Y_blocks]
    dequantized_cb_blocks = [inverse_quantize(block, quantization_table_CbCr, a) for block in quantized_cb_blocks]
    dequantized_cr_blocks = [inverse_quantize(block, quantization_table_CbCr, a) for block in quantized_cr_blocks]

    dezigzag_cb_blocks = [inverse_zigzag(block) for block in quantized_cb_blocks]
    dezigzag_cr_blocks = [inverse_zigzag(block) for block in quantized_cr_blocks]



    idct_cb_blocks = [inverse_dct(block) for block in dezigzag_cb_blocks]
    idct_cr_blocks = [inverse_dct(block) for block in dezigzag_cr_blocks]

    signed_cb_blocks = [to_signed(block) for block in idct_cb_blocks]
    signed_cr_blocks = [to_signed(block) for block in idct_cr_blocks]

    Cb_upsampled = np.concatenate(signed_cb_blocks)
    Cr_upsampled = np.concatenate(signed_cr_blocks)

    Cb_upsampled = Cb_upsampled.reshape(height // 2, width // 2)
    Cr_upsampled = Cr_upsampled.reshape(height // 2, width // 2)

    Cb_upsampled = np.repeat(np.repeat(Cb_upsampled, 2, axis=0), 2, axis=1)
    Cr_upsampled = np.repeat(np.repeat(Cr_upsampled, 2, axis=0), 2, axis=1)

    Y_reconstructed = np.concatenate(dequantized_Y_blocks)

    Y_reconstructed = Y_reconstructed.reshape(height, width)

    Cb_upsampled = Cb_upsampled.astype(np.uint8)
    Cr_upsampled = Cr_upsampled.astype(np.uint8)

    final_image = np.zeros((height, width, 3), dtype=np.uint8)

    final_image[:, :, 0] = Y
    final_image[:, :, 1] = Cb_upsampled
    final_image[:, :, 2] = Cr_upsampled

    final_rgb_image = ycbcr_to_rgb(final_image)

    cv2.imwrite('output.jpg', final_rgb_image)

    return print("\ncompression complete")

