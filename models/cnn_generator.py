# DCGAN-style CNN Generator for Conditional GANs on Fashion-MNIST.

import torch
import torch.nn as nn

class CNNGenerator(nn.Module):
    """
    Conditional DCGAN Generator using transposed convolutions.

    Args:
        z_dim (int): Latent noise dimension.
        embed_dim (int): Label embedding size.
        num_classes (int): Number of Fashion-MNIST classes.
    """
    def __init__(self, z_dim, embed_dim, num_classes):
        super().__init__()
        self.z_dim = z_dim
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.project = nn.Linear(z_dim + embed_dim, 128 * 7 * 7)

        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7x7 → 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),    # 14x14 → 28x28
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_vec = self.label_emb(labels)
        x = torch.cat([z, label_vec], dim=1)
        x = self.project(x).view(-1, 128, 7, 7)
        return self.model(x)
