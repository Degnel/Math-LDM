import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TransformerLM
from llm_dataset import LLMDataset
from constants import PAD_TOKEN
from transformers import get_scheduler

def create_attention_mask(encoded_batch):
    return torch.tensor(
        [
            [1 if token == PAD_TOKEN else 0 for token in sequence]
            for sequence in encoded_batch
        ],
        dtype=torch.bool,
    ).swapaxes(0, 1)


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, targets = batch
        attention_mask = create_attention_mask(inputs)
        outputs = model(inputs, attention_mask=attention_mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM().to(device)
    dataset = LLMDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        loss = train(model, dataloader, optimizer, criterion, device)
        torch.save(model.state_dict(), "./checkpoints/llm_model.pth")
        print(f"Epoch {epoch + 1}, Loss: {loss}")

if __name__ == "__main__":
    main()