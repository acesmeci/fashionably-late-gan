import torch
import torch.nn as nn

class NaiveGenerator(nn.Module):
    def __init__(self, z_dim, embed_dim, num_classes):
        super(NaiveGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.model = nn.Sequential(
            nn.Linear(z_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([z, label_embedding], dim=1)
        out = self.model(x)
        return out.view(out.size(0), 1, 28, 28)
