import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from random import shuffle
from ewc import EWCModule
from alexnet import minMSELoss
from data_loader_light2 import collate_fn

class Autoencoder(EWCModule):
    
    def __init__(self, model, device):
        super(Autoencoder, self).__init__()
        
        self.device = device
        self.encoder = model.features[:6]

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 4, 11, stride=8, output_padding=5)
        )
        
        self.lamdba = 200
        
        self.label = "autoencoder"
        self.to(device)
        
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, return_z=False):
        z = self.encode(x)
        x_ = self.decode(z)
        if return_z:
            return z, x_
        else:
            return x_
        
    def update(self, novelties, familarities, epochs=256):
        
        dataloader = torch.utils.data.DataLoader(novelties, batch_size=16, shuffle=True)
        
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005, weight_decay=1e-3, betas=(0.9,0.999))
        
        with tqdm(range(epochs), desc="Updating autoencoder", leave=None, postfix=dict(loss=0)) as t:
            for epoch in t :
                errors = torch.zeros(1, device=self.device)
                for x in dataloader:
                    x = x[:,0,:,:,:].to(self.device)
                    optimizer.zero_grad()
                    x_ = self(x)
                
                    error = loss(x, x_)
                    error += self.lamdba*self.ewc_loss()
                    errors += error

                    error.backward()
                    optimizer.step()

                t.set_postfix(loss=errors.item()/len(dataloader))

        merged = novelties + familarities
        shuffle(merged)
        self.estimate_fisher(merged)
                
    
    

class Predictor(EWCModule):
    
    def __init__(self, model, device):
        
        super(Predictor, self).__init__()
        
        self.device = device
        
        self.model = model
        self.model.features = self.model.features[6:]
        
        self.lamdba = 700
        
        self.label = "predictor"
        self.to(device)
        
    def forward(self, z):
        return self.model(z)
    

    def update(self, novelties, familarities, epochs=256):

        dataloader = torch.utils.data.DataLoader(novelties, batch_size=16, shuffle=True, collate_fn=collate_fn)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005, weight_decay=1e-3, betas=(0.9,0.999))
        
        with tqdm(range(epochs), desc="Updating predictor", leave=None, postfix=dict(mse=0, ewc=0,loss=0)) as t:
            for epoch in t:
                errors = torch.zeros(1, device=self.device)
                errors_mse = torch.zeros(1, device=self.device)
                errors_ewc = torch.zeros(1, device=self.device)
                for z, y in dataloader:
                    z = z[:,0,:,:,:].to(self.device)
                    y = [yi[0].to(self.device) for yi in y]
                    optimizer.zero_grad()
                    y_pred = self(z)
                
                    error_mse = minMSELoss(y_pred, y, self.device)
                    error_ewc = self.ewc_loss()
                    error = error_mse + self.lamdba*error_ewc

                    errors += error
                    errors_ewc += error_ewc
                    errors_mse += error_mse

                    error.backward()
                    optimizer.step()

                errors_mse = errors_mse.item()/len(dataloader)
                errors_ewc = errors_ewc.item()/len(dataloader)
                errors = errors.item()/len(dataloader)
                t.set_postfix(mse=errors_mse, ewc=errors_ewc, loss=errors)
        
        merged = [n[0] for n in novelties] + [f[0] for f in familarities]
        shuffle(merged)
        self.estimate_fisher(merged)
