import torch
import torch.nn as nn
from constants import VOCAB_SIZE, PAD_TOKEN
import math
from tokenizer import Tokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()

class TransformerLM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerLM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, dropout=dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, attention_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        else:
            attention_mask = None
        output = self.transformer(src, src, src_key_padding_mask=attention_mask)
        output = self.fc_out(output)
        return output
    
    def predict(self, input, max_length=100):
        tokenizer = Tokenizer()
        input_ids = torch.tensor([tokenizer.encode(input)]).to(self.device)
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_ids)
                predictions = outputs.argmax(-1)[:, -1]
                input_ids = torch.cat((input_ids, predictions.unsqueeze(-1)), dim=-1)
                if predictions[-1] == tokenizer.vocab[PAD_TOKEN]:
                    break
        return tokenizer.decode(input_ids.squeeze())