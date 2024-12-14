# import torch
# import torch.nn as nn
# from transformers import GPT2Config, GPT2LMHeadModel

# from constants import PAD_TOKEN, VOCAB_SIZE
# from tokenizer import Tokenizer


# class TransformerLM(nn.Module):
#     def __init__(self, vocab_size=VOCAB_SIZE, d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, dropout=0.1):
#         super(TransformerLM, self).__init__()

#         # Configuration du modèle Hugging Face
#         self.config = GPT2Config(
#             vocab_size=vocab_size,
#             n_positions=10,
#             n_embd=d_model,
#             n_layer=num_layers,
#             n_head=nhead,
#             resid_pdrop=dropout,
#             embd_pdrop=dropout,
#         )

#         # Initialisation d'un modèle Hugging Face sans poids préentraînés
#         self.model = GPT2LMHeadModel(self.config)

#     def forward(self, input_ids, attention_mask=None):
#         """
#         Passe avant du modèle
#         Args:
#             input_ids (Tensor): Séquences d'entrée (batch_size, seq_len).
#             attention_mask (Tensor, optionnel): Masque d'attention (batch_size, seq_len).
#         """
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         return outputs.logits  # Retourne les logits pour le calcul de la loss

#     def predict(self, input_text, max_length=100):
#         """
#         Génération d'une séquence
#         Args:
#             input_text (str): Texte d'entrée.
#             max_length (int): Longueur maximale de la génération.
#         """
#         tokenizer = Tokenizer()
#         input_ids = torch.tensor(tokenizer.encode(input_text)[0]).unsqueeze(0)

#         with torch.no_grad():
#             for _ in range(max_length):
#                 outputs = self.forward(input_ids)
#                 predictions = outputs.argmax(-1)[:, -1]  # Prédiction du prochain token
#                 input_ids = torch.cat((input_ids, predictions.unsqueeze(-1)), dim=-1)

#                 if predictions[-1] == tokenizer.vocab[PAD_TOKEN]:
#                     break

#         return tokenizer.decode(input_ids.squeeze())

#     def load_checkpoint(self, checkpoint_path="./checkpoints/llm_model.pth"):
#         """
#         Chargement des poids depuis un checkpoint.
#         """
#         checkpoint = torch.load(checkpoint_path)
#         self.model.load_state_dict(checkpoint)

import math

import torch
import torch.nn as nn
from torch.nn import Transformer

from constants import PAD_TOKEN, VOCAB_SIZE
from tokenizer import Tokenizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super(PositionalEncoding, self).__init__()
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

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].detach()


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        d_model=6,
        nhead=3,
        num_layers=5,
        dim_feedforward=24,
        dropout=0.1,
    ):
        super(TransformerLM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embeddings = nn.Parameter(torch.empty(VOCAB_SIZE, d_model))
        nn.init.xavier_uniform_(self.embeddings)
        self.pos_encoder = PositionalEncoding(d_model)
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

    def forward(self, tgt, attention_mask=None):
        tgt = torch.matmul(tgt, self.embeddings) * math.sqrt(self.d_model)
        # tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
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
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_ids)
                predictions = outputs.argmax(-1)[:, -1]
                input_ids = torch.cat((input_ids, predictions.unsqueeze(-1)), dim=-1)
                if predictions[-1] == tokenizer.vocab[PAD_TOKEN]:
                    break
        return tokenizer.decode(input_ids.squeeze())

    def load_checkpoint(self, checkpoint_path="./checkpoints/llm_model.pth"):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.load_state_dict(checkpoint)
