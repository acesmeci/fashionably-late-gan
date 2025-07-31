# Discriminator with both Feature Matching and Minibatch Discrimination.
# Outputs both logits and internal features when requested.

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator with Feature Matching + Minibatch Discrimination.

    Args:
        embed_dim (int): Label embedding size.
        num_classes (int): Number of classes.
    """
    def __init__(self, embed_dim, num_classes):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.feature_net = nn.Sequential(
            nn.Linear(784 + embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(256 + 1, 1),  # +1 for minibatch stddev
            nn.Sigmoid()
        )

    def forward(self, x, labels, return_features=False):
        """
        Args:
            x (Tensor): Input image (batch_size, 1, 28, 28)
            labels (Tensor): Class labels
            return_features (bool): If True, also returns feature layer output.

        Returns:
            Tensor: Discriminator output
            Optional[Tensor]: Feature representation before final layer
        """
        x = x.view(x.size(0), -1)
        label_embedding = self.label_emb(labels)
        x = torch.cat([x, label_embedding], dim=1)

        features = self.feature_net(x)

        # Add minibatch stddev
        stddev = features.std(dim=0).mean().unsqueeze(0).expand(x.size(0), 1)
        combined = torch.cat([features, stddev], dim=1)

        out = self.output_layer(combined)

        if return_features:
            return out, features
        else:
            return out
