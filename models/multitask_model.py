import torch
from transformers import AutoModel, AutoTokenizer

class SentenceTransformer(torch.nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, sentences):
        tokens = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        outputs = self.encoder(**tokens)
        return outputs.last_hidden_state.mean(dim=1)  # Mean Pooling

class MultiTaskModel(SentenceTransformer):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier_A = torch.nn.Linear(hidden_size, 3)  # Sentence Classification
        self.classifier_B = torch.nn.Linear(hidden_size, 2)  # Sentiment Analysis

    def forward(self, sentences, task='A'):
        embeddings = super().forward(sentences)
        if task == 'A':
            return self.classifier_A(embeddings)
        else:
            return self.classifier_B(embeddings)