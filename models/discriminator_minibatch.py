import torch
import torch.nn as nn

class Discriminator(nn.Module):
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
            nn.Linear(256 + 1, 1),  # +1 for stddev feature
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        label_embedding = self.label_emb(labels)
        x = torch.cat([x, label_embedding], dim=1)

        features = self.feature_net(x)

        # Add minibatch standard deviation (scalar)
        stddev = features.std(dim=0).mean().unsqueeze(0).expand(x.size(0), 1)
        x = torch.cat([features, stddev], dim=1)

        return self.output_layer(x)
