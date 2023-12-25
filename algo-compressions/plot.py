import matplotlib.pyplot as plt

def plots():
    functions = ["HA", "AC", "LZ78", "rle_bwt_mtf_rle_ha", "bwt_mtf_ha", "bwt_mtf_ac", "rle_bwt_mtf_rle_ac"]
    execution_times = [0.9953, 4.001, 0.9982, 34.1608, 10.0915, 11.0023, 30.9998]
    compressed_sizes = [16487, 6024, 7616, 8094, 7245, 6800, 5736]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.barh(functions, execution_times, color='blue')
    plt.xlabel('Время выполнения (миллисекунды)')
    plt.title('Время выполнения функций')

    plt.subplot(1, 2, 2)
    plt.barh(functions, compressed_sizes, color='green')
    plt.xlabel('Размер (биты)')
    plt.title('Размер сжатых данных')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    plt.show()
