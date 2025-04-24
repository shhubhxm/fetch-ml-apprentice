import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from models.multitask_model import MultiTaskModel
from data.dummy_data import get_dummy_data

if __name__ == "__main__":
    model = MultiTaskModel()
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = CrossEntropyLoss()

    data = get_dummy_data()

    for epoch in range(2):
        print(f"\n--- Epoch {epoch+1} ---")
        for sentence, label, task in data:
            optimizer.zero_grad()
            logits = model([sentence], task=task)
            loss = loss_fn(logits, torch.tensor([label]))
            loss.backward()
            optimizer.step()

            print(f"Task {task} | Loss: {loss.item():.4f}")
