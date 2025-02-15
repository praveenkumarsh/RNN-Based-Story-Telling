from transformers import AutoTokenizer, AutoModel
import torch

class TextEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def clean_sentence(output, idx2word):
    words = [idx2word[i] for i in output if i not in (0, 1)]  # Remove <start> and <end>
    return ' '.join(words).capitalize()