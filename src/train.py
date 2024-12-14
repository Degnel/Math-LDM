import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import get_scheduler

from constants import PAD_TOKEN
from llm_dataset import LLMDataset
from ldm_dataset import LDMDataset
from model import TransformerLM
import torch.nn.functional as F

class CrossEntropySoftLabels(torch.nn.Module):
    def forward(self, logits, targets):
        log_softmax_logits = F.log_softmax(logits, dim=1)
        loss = -torch.sum(targets * log_softmax_logits) / (targets.shape[0] * targets.shape[1])
        return loss

def create_attention_mask(encoded_batch):
    return torch.tensor(
        [
            [1 if token == PAD_TOKEN else 0 for token in sequence]
            for sequence in encoded_batch
        ],
        dtype=torch.bool,
    )


def train(model, dataloader, optimizer, criterion, mode):
    model.train()
    total_loss = 0
    for batch in dataloader:
        if mode == 'llm':
            inputs, targets = batch
            attention_mask = create_attention_mask(inputs)
            inputs = F.one_hot(inputs).to(dtype=torch.float32)
            outputs = model(inputs, attention_mask=attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        elif mode == 'ldm':
            inputs, targets, mask, step = batch
            outputs = model(inputs, step=step)
            loss = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(dataloader)

def main(mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "ldm":
        dataset = LDMDataset()
        criterion = CrossEntropySoftLabels()
        model = TransformerLM(max_steps=100).to(device)
    elif mode == "llm":
        dataset = LLMDataset()
        criterion = torch.nn.CrossEntropyLoss()
        model = TransformerLM().to(device)

    subset_size = 100
    total_length = len(dataset)
    indices = torch.randperm(total_length)[:subset_size]
    dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(5):
        loss = train(model, dataloader, optimizer, criterion, mode)
        torch.save(model.state_dict(), f"./checkpoints/{mode}_model.pth")
        print(f"Epoch {epoch + 1}, Loss: {loss}")