import torch

from constants import ALPHABET, PAD_TOKEN


class Tokenizer:
    def __init__(self):
        self.vocab = {char: idx for idx, char in enumerate(ALPHABET)}
        self.vocab[PAD_TOKEN] = len(self.vocab)
        self.inv_vocab = {idx: char for char, idx in self.vocab.items()}
        self.equal_token_id = self.vocab["="]
        self.pad_token_id = self.vocab[PAD_TOKEN]

    def encode(self, sequence):
        if isinstance(sequence, str):
            sequence = [sequence]
        return [
            torch.tensor([self.vocab[char] for char in seq], dtype=torch.long)
            for seq in sequence
        ]

    def decode(self, tokens):
        return " ".join([self.inv_vocab[token.item()] for token in tokens])
