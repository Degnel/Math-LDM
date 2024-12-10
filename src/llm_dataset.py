import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

from dataset import MathExpressionDataset
from tokenizer import Tokenizer

# Faire en sorte que comme MathExpressionDataset, LLMDataset enregistre le fichier une fois le calcul effectué


class LLMDataset(MathExpressionDataset):
    def __init__(
        self,
        num_samples=100000,
        file_path="./data/dataset.txt",
        force_recreate=False,
        max_length=10,
    ):
        super().__init__(num_samples, file_path, force_recreate)
        tokenizer = Tokenizer()
        self.tokens = tokenizer.encode(self.data)
        self.llm_data = self._create_shifted_dataset(tokenizer, max_length)

    def _create_shifted_dataset(self, tokenizer: Tokenizer, max_length: int):
        dataset = []
        for sample in self.tokens:
            equal_index = (
                (sample == tokenizer.equal_token_id).nonzero(as_tuple=True)[0].item()
            )

            for i in range(equal_index + 1, len(sample) + 1):
                input_seq = self.get_tensor_with_default(
                    sample, i - max_length, i, tokenizer.pad_token_id
                )
                output_seq = self.get_tensor_with_default(
                    sample, i + 1 - max_length, i + 1, tokenizer.pad_token_id
                )
                dataset.append((input_seq, output_seq))

        return dataset

    def get_tensor_with_default(self, tens, lower, upper, default_value=0):
        indices = range(lower, upper)
        tens_size = tens.size(0)
        valid_indices = [
            (tens[idx] if 0 <= idx < tens_size else default_value) for idx in indices
        ]
        return torch.tensor(valid_indices, dtype=tens.dtype)

    def __len__(self):
        return len(self.llm_data)

    def __getitem__(self, idx):
        return self.llm_data[idx]
