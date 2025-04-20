import os
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import torch
import random
from PIL import Image

dataset_dir = r'C:\Users\akrivacic\Downloads\sport-classification'

output_dir = r'C:\Users\akrivacic\Downloads\augmented_sports_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Nasumična horizontalna rotacija
    transforms.RandomRotation(30),  # Rotacija za nasumične kutove
    transforms.RandomResizedCrop(224),  # Nasumično izrezivanje na 224x224
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Promjena boje
    transforms.RandomVerticalFlip(),  # Nasumična vertikalna rotacija
    transforms.ToTensor(),  # Pretvaranje u tensor
])

dataset = datasets.ImageFolder(root=dataset_dir, transform=data_transforms)

sampled_images = random.sample(range(len(dataset)), int(len(dataset) * 0.2))
sampled_images_dataset = [dataset[i] for i in sampled_images]

sampled_loader = DataLoader(sampled_images_dataset, batch_size=32, shuffle=True)

to_pil = ToPILImage()

def save_augmented_images(sampled_loader, output_dir):
    for i, (images, labels) in enumerate(sampled_loader):
        for j, image in enumerate(images):
            # Pretvaranje tensora u PIL sliku
            image_pil = to_pil(image)
            # Definiranje imena datoteke na temelju indeksa batch-a i indeksa slike
            image_filename = f"augmented_image_{i * 32 + j}.jpg"
            # Spremanje slike na disk
            image_pil.save(os.path.join(output_dir, image_filename))
            print(f"Spremio sliku: {image_filename}")

save_augmented_images(sampled_loader, output_dir)

print("Spremanje proširenih slika završeno!")


