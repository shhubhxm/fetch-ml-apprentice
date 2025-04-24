import torch
from torch.optim import Adam
from models.multitask_model import MultiTaskModel
from data.dummy_data import get_dummy_data

def train():
    model = MultiTaskModel()
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    data = get_dummy_data()
    model.train()
    for epoch in range(3):
        for sentence, label, task in data:
            optimizer.zero_grad()
            output = model([sentence], task=task)
            loss = criterion(output, torch.tensor([label]))
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Task {task}, Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train()