# Improved MLP Generator with BatchNorm for Conditional GANs.
# Uses architectural stabilization tricks from the DCGAN paper.

import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    MLP Generator with BatchNorm (Stable variant).

    Args:
        z_dim (int): Latent vector size.
        embed_dim (int): Label embedding size.
        num_classes (int): Number of Fashion-MNIST classes.
    """
    def __init__(self, z_dim, embed_dim, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.model = nn.Sequential(
            nn.Linear(z_dim + embed_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([z, label_embedding], dim=1)
        out = self.model(x)
        return out.view(out.size(0), 1, 28, 28)
