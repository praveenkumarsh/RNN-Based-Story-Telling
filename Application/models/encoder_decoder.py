import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights 

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        return self.embed(features)

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), 
                      torch.zeros(1, 1, hidden_size))

    def forward(self, features, captions):
        cap_embedding = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), dim=1)
        lstm_out, self.hidden = self.lstm(embeddings)
        return self.linear(lstm_out)

    def sample(self, inputs, states=None, max_len=20):
        res = []
        for _ in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.linear(lstm_out.squeeze(1))
            _, predicted = outputs.max(1)
            res.append(predicted.item())
            if predicted == 1:  # <end> token
                break
            inputs = self.embed(predicted).unsqueeze(1)
        return res