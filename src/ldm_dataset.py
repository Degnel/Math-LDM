import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

from dataset import MathExpressionDataset
from tokenizer import Tokenizer
import torch.nn.utils.rnn as rnn_utils
from torch.nn import functional as F

# Faire en sorte que comme MathExpressionDataset, LLDDataset enregistre le fichier une fois le calcul effectué


class LDMDataset(MathExpressionDataset):
    def __init__(
        self,
        num_samples=100000,
        file_path="./data/dataset.txt",
        force_recreate=False,
        max_length=16,
        diffusion_steps=100,
    ):
        super().__init__(num_samples, file_path, force_recreate)
        tokenizer = Tokenizer()
        self.tokens = tokenizer.encode(self.data)
        self.tokens = rnn_utils.pad_sequence(
            self.tokens, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        self.llm_data = self._create_diffused_dataset(
            tokenizer, max_length, diffusion_steps
        )

    def _create_diffused_dataset(
        self, tokenizer: Tokenizer, max_length: int, diffusion_steps: int
    ):
        size = len(self.tokens)
        dataset = [None] * diffusion_steps * size
        equal_mask = self.tokens == tokenizer.equal_token_id
        equal_indices = equal_mask.nonzero(as_tuple=True)[1]
        cols = torch.arange(equal_mask.size(1))
        mask = (cols.unsqueeze(0) > equal_indices.unsqueeze(1)).unsqueeze(-1)

        # L'input_seq correspond à la version noisy
        # L'output_seq correspond à la version clean
        input_seq = F.one_hot(self.tokens).to(dtype=torch.float32)
        output_seq = input_seq.clone()
        for step in range(diffusion_steps):
            dataset[step * size : (step + 1) * size] = zip(
                input_seq, output_seq, mask, (step,) * size
            )

            output_seq = input_seq.clone()
            input_seq += torch.randn_like(input_seq) * mask
            masked_input_seq = F.softmax(input_seq * mask, dim=-1)
            input_seq = input_seq * (~mask) + masked_input_seq * mask

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