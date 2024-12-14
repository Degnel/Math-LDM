import math

import torch
import torch.nn as nn
from torch.nn import Transformer

from constants import PAD_TOKEN, VOCAB_SIZE
from tokenizer import Tokenizer


import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16, max_steps=None):
        super(PositionalEncoding, self).__init__()
        
        # Positional embedding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        
        # Temporal embedding
        self.max_steps = max_steps
        if max_steps is not None:
            te = torch.zeros(max_steps, d_model)
            step = torch.arange(0, max_steps).unsqueeze(1).float()
            te[:, 0::2] = torch.sin(step * div_term)
            te[:, 1::2] = torch.cos(step * div_term)
            te = te.unsqueeze(0)
            self.register_buffer("te", te)
        else:
            self.te = None

    def forward(self, x, t=None):
        """
        Arguments:
        - x: Input tensor of shape (batch_size, seq_len, d_model)
        - t: (Optional) Current timestep for temporal embedding

        Returns:
        - Tensor of shape (batch_size, seq_len, d_model) with positional (and optionally temporal) embeddings added.
        """
        # Add positional encoding
        x = x + self.pe[:, :x.size(1)].detach()
        
        # Add temporal encoding if conditions are met
        if t is not None and self.max_steps is not None:
            if t.dim() != 1 or t.size(0) != x.size(0):
                raise ValueError(f"Expected t to have shape (batch_size,), but got {t.shape}")
            
            if not torch.all((0 <= t) & (t < self.max_steps)):
                raise ValueError(f"All values in t must be in range [0, {self.max_steps}).")
            
            # Gather the temporal embeddings for each batch element
            batch_temporal_embeddings = self.te[0, t, :]  # Shape: (batch_size, d_model)
            batch_temporal_embeddings = batch_temporal_embeddings.unsqueeze(1)
            
            # Add temporal embeddings to all time steps of the sequence
            x = x + batch_temporal_embeddings.expand(-1, x.size(1), -1)
        
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        d_model=6,
        nhead=3,
        num_layers=5,
        dim_feedforward=24,
        dropout=0.1,
        max_len=16,
        max_steps=None,
    ):
        super(TransformerLM, self).__init__()

        self.mode = "llm" if max_steps is None else "ldm"
        self.max_len = max_len
        self.max_steps = max_steps
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embeddings = nn.Parameter(torch.empty(VOCAB_SIZE, d_model))
        nn.init.xavier_uniform_(self.embeddings)
        self.pos_encoder = PositionalEncoding(d_model, max_len, max_steps)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, tgt, attention_mask=None, step=None):
        tgt = torch.matmul(tgt, self.embeddings) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt, step)
        seq_len = tgt.size(1)
        causal_mask = Transformer.generate_square_subsequent_mask(seq_len)
        output = self.transformer_decoder(
            tgt=tgt,
            memory=torch.zeros_like(tgt),
            tgt_mask=causal_mask,
            tgt_key_padding_mask=attention_mask,
        )

        output = self.fc_out(output)
        return output

    def predict(self, input, max_length=100):
        tokenizer = Tokenizer()
        input_ids = torch.tensor(tokenizer.encode(input)[0]).unsqueeze(0)
        if self.mode == "llm":
            with torch.no_grad():
                for _ in range(max_length):
                    outputs = self.forward(input_ids)
                    predictions = outputs.argmax(-1)[:, -1]
                    input_ids = torch.cat((input_ids, predictions.unsqueeze(-1)), dim=-1)
                    if predictions[-1] == tokenizer.vocab[PAD_TOKEN]:
                        break
            return tokenizer.decode(input_ids.squeeze())
        elif self.mode == "ldm":
            one_hot = F.one_hot(input_ids, VOCAB_SIZE)
            rand = torch.randn((self.max_len - one_hot.size(1), one_hot.size(2))).unsqueeze(0)
            x = torch.cat((one_hot, rand), dim=1)
            for step in range(self.max_steps):
                x[:, one_hot.size(1):] = self.forward(x, step=torch.tensor([step]))[:, one_hot.size(1):]

            outputs = x.argmax(-1)
            return tokenizer.decode(input_ids.squeeze())

    def load_checkpoint(self, mode, checkpoint_path="./checkpoints"):
        checkpoint = torch.load(checkpoint_path+f'/{mode}_model.pth', weights_only=True)
        self.load_state_dict(checkpoint)