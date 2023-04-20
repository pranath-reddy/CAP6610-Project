import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np
from torchvision.datasets import CelebA
from torch.utils.data import Dataset

# Set device
device = torch.device("mps")

# Hyperparameters
lr = 0.0002
num_epochs = 200
latent_dim = 100

# CelebA dataset
batch_size = 128
image_size = 64
channels = 3
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Wrapper for limiting dataset size
class LimitedDataset(Dataset):
    def __init__(self, original_dataset, limit):
        self.original_dataset = original_dataset
        self.limit = limit

    def __getitem__(self, index):
        return self.original_dataset[index]

    def __len__(self):
        return min(self.limit, len(self.original_dataset))

# Set a limit on the number of samples
samples_limit = 20000

# Create the CelebA dataset
celeba_data = datasets.ImageFolder(root='./data/', transform=transform)

# Wrap the dataset with the LimitedDataset class
limited_celeba_data = LimitedDataset(celeba_data, samples_limit)

# Create the DataLoader with the limited dataset
data_loader = DataLoader(limited_celeba_data, batch_size=batch_size, shuffle=True)

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, latent_dim * 2, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x)

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            )
        
    def forward(self, x):
        return self.main(x)
        
encoder = Encoder().to(device)
decoder = Decoder().to(device)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, betas=(0.5, 0.999))
reconstruction_loss = nn.MSELoss(reduction='sum')

# Training Loop
Loss = []
fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
for epoch in tqdm(range(num_epochs)):
    loss_epoch = 0
    num_batches = 0
    for i, (images, _) in enumerate(data_loader):
        current_batch_size = images.size(0)
        images = images.to(device)
        # Encode the images
        mu_logvar = encoder(images).view(-1, 2, latent_dim)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]

        # Sample from the latent space using the reparameterization trick
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std).to(device)
        latent_space = mu + std * z

        # Decode the samples
        reconstructed_images = decoder(latent_space.view(-1, latent_dim, 1, 1))

        # Compute the VAE loss
        rec_loss = reconstruction_loss(reconstructed_images, images) / current_batch_size
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / current_batch_size
        vae_loss = rec_loss + kl_divergence

        # Update the weights
        optimizer.zero_grad()
        vae_loss.backward()
        optimizer.step()

        loss_epoch += vae_loss.item()
        num_batches += 1

    # Compute average epoch loss
    loss_epoch /= num_batches
    Loss.append(loss_epoch)

    # Generate and save sample images
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fixed_fake_images = decoder(fixed_noise)
            fixed_fake_images = fixed_fake_images.view(fixed_fake_images.size(0), channels, image_size, image_size)
            vutils.save_image(fixed_fake_images.data, f"./Samples/VAE_Faces/CelebA_samples_epoch_{epoch+1}.png", nrow=16, normalize=True)

# Save Loss
np.save('./Plots/VAE_Loss_Faces.npy', Loss)

# Save Weights
torch.save(encoder.state_dict(), "./Weights/CelebA_VAE_encoder.pth")
torch.save(decoder.state_dict(), "./Weights/CelebA_VAE_decoder.pth")


