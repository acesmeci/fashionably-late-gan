"""
Training script for the stabilized MLP-based Conditional GAN on Fashion-MNIST.
Includes classic GAN stabilization tricks: BatchNorm, LeakyReLU, and low-momentum Adam.
"""

import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from models.generator import Generator
from models.discriminator import Discriminator

# Hyperparameters
batch_size = 64
z_dim = 100
lr = 2e-4
epochs = 10
embedding_dim = 10
image_size = 28
num_classes = 10

# Create output directory for baseline model
output_dir = "samples/mlp_stable"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset: Fashion-MNIST (normalized to [-1, 1] for Tanh)
data = datasets.FashionMNIST(
    root="./data",
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

# Initialize models
G = Generator(z_dim, embedding_dim, num_classes).to(device)
D = Discriminator(embedding_dim, num_classes).to(device)

# Loss and Optimizers
criterion = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Fixed noise and labels for consistent visual output
fixed_noise = torch.randn(10, z_dim).to(device)
fixed_labels = torch.arange(0, 10).to(device)

# Training loop
for epoch in range(epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        real_images = real_images.to(device)
        labels = labels.to(device)
        batch_size = real_images.size(0)

        real_targets = torch.ones(batch_size, 1).to(device)
        fake_targets = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = G(z, labels)

        real_preds = D(real_images, labels)
        fake_preds = D(fake_images.detach(), labels)

        d_loss = criterion(real_preds, real_targets) + criterion(fake_preds, fake_targets)
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = G(z, labels)
        preds = D(fake_images, labels)

        g_loss = criterion(preds, real_targets)
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f"[Epoch {epoch+1}/{epochs}]  D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # Save fixed generated grid for this epoch
    with torch.no_grad():
        generated = G(fixed_noise, fixed_labels)
        grid = make_grid(generated, nrow=10, normalize=True)
        save_image(grid, os.path.join(output_dir, f"epoch_{epoch+1}.png"))
