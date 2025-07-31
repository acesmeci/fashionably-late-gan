# DCGAN-style CNN Discriminator for Conditional GANs.

import torch
import torch.nn as nn

class CNNDiscriminator(nn.Module):
    """
    Conditional DCGAN Discriminator.

    Args:
        embed_dim (int): Size of label embedding.
        num_classes (int): Number of Fashion-MNIST classes.
    """
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.model = nn.Sequential(
            nn.Conv2d(1 + 1, 64, kernel_size=4, stride=2, padding=1),  # 28x28 → 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),     # 14x14 → 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        label_img = self.label_emb(labels).unsqueeze(2).unsqueeze(3)  # shape: (B, embed_dim, 1, 1)
        label_img = label_img.expand(-1, 1, 28, 28)                   # broadcast to image shape
        x = torch.cat([x, label_img], dim=1)                          # concat channel-wise
        return self.model(x)
