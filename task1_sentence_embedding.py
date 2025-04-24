from models.multitask_model import SentenceTransformer

if __name__ == "__main__":
    model = SentenceTransformer()
    sentences = ["The cat sits outside.", "The sun is shining brightly."]
    embeddings = model(sentences)

    print("Embedding shape:", embeddings.shape)
    print("Embeddings:", embeddings)
