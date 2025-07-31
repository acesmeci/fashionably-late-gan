# Improved MLP Discriminator with LeakyReLU and stable architecture.

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Stable MLP Discriminator with LeakyReLU.

    Args:
        embed_dim (int): Label embedding size.
        num_classes (int): Number of Fashion-MNIST classes.
    """
    def __init__(self, embed_dim, num_classes):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.model = nn.Sequential(
            nn.Linear(784 + embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        label_embedding = self.label_emb(labels)
        x = torch.cat([x, label_embedding], dim=1)
        return self.model(x)
