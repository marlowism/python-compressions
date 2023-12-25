from decimal import Decimal, getcontext

class ArithmeticEncoder:
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        self.blocks = [data[i:i + block_size] for i in range(0, len(data), block_size)]
        self.encoded_blocks = []

    def calculate_frequency(self, block):
        frequency_table = {}
        for char in block:
            if char in frequency_table:
                frequency_table[char] += 1
            else:
                frequency_table[char] = 1
        return frequency_table

    def encode_block(self, block):
        getcontext().prec = 20
        frequency_table = self.calculate_frequency(block)
        sorted_symbols = sorted(frequency_table.keys())

        probability_table = {}
        low = Decimal(0)
        for symbol in sorted_symbols:
            high = low + Decimal(frequency_table[symbol]) / Decimal(len(block))
            probability_table[symbol] = (low, high)
            low = high

        interval_low = Decimal(0)
        interval_high = Decimal(1)

        for char in block:
            symbol_range = probability_table[char]
            interval_range = interval_high - interval_low
            interval_high = interval_low + interval_range * symbol_range[1]
            interval_low = interval_low + interval_range * symbol_range[0]

        encoded_value = (interval_low + interval_high) / Decimal(2)
        self.encoded_blocks.append(encoded_value)
        self.probability_table = probability_table

    def encode(self):
        for block in self.blocks:
            self.encode_block(block)

        encoded = "".join(str(encoded_block) for encoded_block in self.encoded_blocks)
        return encoded

    def save_probability_table(self, filename):
        with open(filename, "w") as file:
            for symbol, (low, high) in self.probability_table.items():
                file.write(f"Symbol: {symbol}, Probability : ({low}, {high})\n")
