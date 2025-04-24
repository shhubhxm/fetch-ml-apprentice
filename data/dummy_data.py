import torch

def get_dummy_data():
    sentences = ["The product is amazing", "Service was okay", "Absolutely terrible experience"]
    labels_A = torch.tensor([2, 1, 0])  # Task A: Sentiment levels
    labels_B = torch.tensor([1, 1, 0])  # Task B: Binary sentiment
    tasks = ['A', 'A', 'B']
    return list(zip(sentences, labels_A.tolist(), tasks))