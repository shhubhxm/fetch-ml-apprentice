from models.multitask_model import MultiTaskModel

if __name__ == "__main__":
    model = MultiTaskModel()
    model.eval()

    sentences = ["Great service!", "Terrible experience."]
    
    output_A = model(sentences, task='A')
    output_B = model(sentences, task='B')

    print("Output for Task A (Classification):", output_A)
    print("Output for Task B (Sentiment):", output_B)
