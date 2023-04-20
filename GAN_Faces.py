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

# wrapper
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

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
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

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
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
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop
Gen_Loss = []
Disc_Loss = []
fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
for epoch in tqdm(range(num_epochs)):
    gen_loss_epoch = 0
    disc_loss_epoch = 0
    num_batches = 0
    for i, (images, _) in enumerate(data_loader):
        current_batch_size = images.size(0)
        images = images.to(device)
        real_labels = torch.ones(current_batch_size, device=device)
        fake_labels = torch.zeros(current_batch_size, device=device)

        # Train discriminator
        optimizer_d.zero_grad()
        real_outputs = discriminator(images)
        real_loss = criterion(real_outputs, real_labels)
        real_score = real_outputs

        noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        fake_outputs = discriminator(fake_images.detach())
        fake_loss = criterion(fake_outputs, fake_labels)
        fake_score = fake_outputs

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()

        disc_loss_epoch += d_loss.item()
        num_batches += 1

        # Train generator
        optimizer_g.zero_grad()
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

        gen_loss_epoch += g_loss.item()

    # Compute average epoch losses
    disc_loss_epoch /= num_batches
    gen_loss_epoch /= num_batches
    Disc_Loss.append(disc_loss_epoch)
    Gen_Loss.append(gen_loss_epoch)

    # Generate and save sample images
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fixed_fake_images = generator(fixed_noise)
            fixed_fake_images = fixed_fake_images.view(fixed_fake_images.size(0), channels, image_size, image_size)
            vutils.save_image(fixed_fake_images.data, f"./Samples/GAN_Faces/CelebA_samples_epoch_{epoch+1}.png", nrow=16, normalize=True)

# Save Loss
np.save('./Plots/Disc_Loss_Faces.npy', Disc_Loss)
np.save('./Plots/Gen_Loss_Faces.npy', Gen_Loss)

# Save Weights
torch.save(generator.state_dict(), "./Weights/CelebA_GAN_generator.pth")
torch.save(discriminator.state_dict(), "./Weights/CelebA_GAN_discriminator.pth")