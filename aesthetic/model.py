from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm

from os.path import join
#from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip


from PIL import Image, ImageFile


#####  This script will predict the aesthetic score for this image file


# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class aesthetic_predictor(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
        s = torch.load("aesthetic/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
        self.model.load_state_dict(s)
        self.model.to(self.device)
        self.model.eval()
        model2, preprocess = clip.load("ViT-L/14", device=self.device)  #RN50x64   
        self.model2 = model2
        self.preprocess = preprocess
    def predict(self, img):
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
           image_features = self.model2.encode_image(image)
        
        im_emb_arr = normalized(image_features.cpu().detach().numpy() )
        
        prediction = self.model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
        return prediction
    def predict_2(self, image):
        with torch.no_grad():
            image_features = self.model2.encode_image(image)
            
            im_emb_arr = normalized(image_features.cpu().detach().numpy() )
            
            prediction = self.model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
        return prediction
