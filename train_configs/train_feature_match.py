import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from models.generator import Generator
from models.discriminator_featurematch import Discriminator

# Hyperparameters
batch_size = 64
z_dim = 100
lr = 2e-4
epochs = 10
embedding_dim = 10
num_classes = 10

output_dir = "samples/feature_match"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = datasets.FashionMNIST(
    root="./data",
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

G = Generator(z_dim, embedding_dim, num_classes).to(device)
D = Discriminator(embedding_dim, num_classes).to(device)

bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()

g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

fixed_noise = torch.randn(10, z_dim).to(device)
fixed_labels = torch.arange(0, 10).to(device)

for epoch in range(epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        real_images = real_images.to(device)
        labels = labels.to(device)
        batch_size = real_images.size(0)

        real_targets = torch.ones(batch_size, 1).to(device)
        fake_targets = torch.zeros(batch_size, 1).to(device)

        # === Train Discriminator ===
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = G(z, labels)

        real_preds, _ = D(real_images, labels, return_features=True)
        fake_preds, _ = D(fake_images.detach(), labels, return_features=True)

        d_loss = bce_loss(real_preds, real_targets) + bce_loss(fake_preds, fake_targets)
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # === Train Generator (feature matching) ===
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = G(z, labels)

        _, real_feats = D(real_images, labels, return_features=True)
        _, fake_feats = D(fake_images, labels, return_features=True)

        # Match feature means
        g_loss = mse_loss(fake_feats.mean(dim=0), real_feats.mean(dim=0))

        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f"[Epoch {epoch+1}/{epochs}]  D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    with torch.no_grad():
        generated = G(fixed_noise, fixed_labels)
        grid = make_grid(generated, nrow=10, normalize=True)
        save_image(grid, os.path.join(output_dir, f"epoch_{epoch+1}.png"))
