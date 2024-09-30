import os
import numpy as np
import shutil
from sklearn.cluster import DBSCAN
import torch

import random
import numpy as np
import torch
import torch.nn as nn

import torchvision.transforms as transforms

import concurrent.futures
from PIL import Image

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose([
    transforms.Resize((288,288)),
    transforms.ToTensor(),
    normalize,
])
skew_320 = transforms.Compose([
    transforms.Resize([320, 320]),
    transforms.ToTensor(),
    normalize,
])

def just_transform(img, transform, device):
    img_tensor = transform(img).to(device)
    return img_tensor


def process_images(images, transform, device, num_workers=16):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(just_transform, image, transform, device): idx for idx, image in enumerate(images)}
        transformed_images = [None] * len(images)
        
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            transformed_images[idx] = future.result()
            
    return torch.stack(transformed_images)


def get_SSCD_feature(images, model, device):
    with torch.no_grad():
        images = process_images(images, small_288, device, num_workers = 16)
        feats = model(images)
        feats = nn.functional.normalize(feats, dim=1, p=2)
    return feats

def cluster_images(sim_matrix, threshold, min_samples, num_images):
    distance_matrix = 1 - sim_matrix
    distance_matrix[distance_matrix < 0] = 0

    clustering = DBSCAN(eps=threshold, min_samples=min_samples, metric='precomputed').fit(distance_matrix)
    labels = clustering.labels_

    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # -1 is for noise

    return_array = []
    
    if unique_labels:
        for cluster_label in unique_labels:
            cluster_indices = np.where(labels == cluster_label)[0]
            return_array.append(cluster_indices.tolist())
    return return_array