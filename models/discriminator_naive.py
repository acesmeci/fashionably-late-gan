# Naive MLP Discriminator for Conditional GAN.
# Implements a basic discriminator without minibatch features or feature matching.

import torch
import torch.nn as nn

class NaiveDiscriminator(nn.Module):
    """
    Naive Conditional Discriminator (MLP).

    Args:
        embed_dim (int): Size of label embeddings.
        num_classes (int): Number of target classes.
    """
    def __init__(self, embed_dim, num_classes):
        super(NaiveDiscriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.model = nn.Sequential(
            nn.Linear(784 + embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        """
        Forward pass of the discriminator.

        Args:
            x (Tensor): Input image (batch_size, 1, 28, 28)
            labels (Tensor): Class labels
        
        Returns:
            Tensor: Discriminator output (batch_size, 1)
        """
        x = x.view(x.size(0), -1)
        label_embedding = self.label_emb(labels)
        x = torch.cat([x, label_embedding], dim=1)
        return self.model(x)
