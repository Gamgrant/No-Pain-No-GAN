import os
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
import numpy as np

def calculate_fid(act1, act2):
    """Calculate the FID score between two batches of activations."""
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def get_activations(dataloader, model, device):
    """Get activations for a dataset of images using a specified model."""
    model.eval()
    features = []
    with torch.no_grad():
        for data in dataloader:
            images, _ = data
            images = images.to(device)
            pred = model(images)[0]
            features.append(pred.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

def main(real_images_path, fake_images_path, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    real_dataset = ImageFolder(root=real_images_path, transform=transform)
    fake_dataset = ImageFolder(root=fake_images_path, transform=transform)
    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    fake_dataloader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)
    
    model = inception_v3(pretrained=True).to(device)
    model.fc = torch.nn.Identity()
    
    real_activations = get_activations(real_dataloader, model, device)
    fake_activations = get_activations(fake_dataloader, model, device)
    
    fid_score = calculate_fid(real_activations, fake_activations)
    print(f"FID score: {fid_score}")

if __name__ == "__main__":
    REAL_IMAGES_PATH = "path/to/real/images"
    FAKE_IMAGES_PATH = "path/to/generated/images"
    main(REAL_IMAGES_PATH, FAKE_IMAGES_PATH)
