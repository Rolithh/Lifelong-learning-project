import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models import Autoencoder, Predictor
from alexnet import load_model, minMSELoss


mse = nn.MSELoss()

class CLeaR:
    
    def __init__(self):
        
        model = load_model()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.autoencoder = Autoencoder(model, self.device)
        self.predictor = Predictor(model, self.device)
        
        
        
        self.buffer_autoencoder_f = []
        self.buffer_autoencoder_n = []
        self.lim_autoencoder = 4000
        self.threshold_autoencoder = 1
        
        self.buffer_predictor_f = []
        self.buffer_predictor_n = []
        self.lim_predictor = 4000
        self.threshold_predictor = 1
        
        self.alpha = 0.95
        
          
        
    def train_on_task(self, dataloader, warmup=False):
        
        
        for x, y in tqdm(dataloader, desc="Reading data stream", leave=None):

            x = x.to(self.device)
            y = [yi.to(self.device) for yi in y]
            
            
            with torch.no_grad():
                z, x_ = self.autoencoder(x, True)
                error_autoencoder = mse(x, x_)
            
            if error_autoencoder > self.threshold_autoencoder or warmup:
                self.buffer_autoencoder_n.append(x)
            else:
                self.buffer_autoencoder_f.append(x)
                
            with torch.no_grad():
                y_pred = self.predictor(z)
                error_predictor = minMSELoss(y_pred, y)
            
            if error_predictor > self.threshold_predictor or warmup:
                self.buffer_predictor_n.append((z, y))
            else:
                self.buffer_predictor_f.append((z, y))
            
            
            if len(self.buffer_autoencoder_n) >= self.lim_autoencoder and not warmup:
                self.update_autoencoder()
                
            if len(self.buffer_predictor_n) >= self.lim_predictor and not warmup:
                self.update_predictor()

        self.update_autoencoder()     
        self.update_predictor()              
                
            
    def update_autoencoder(self):

        self.autoencoder.update(self.buffer_autoencoder_n, self.buffer_autoencoder_f)
            
        with torch.no_grad():
            x = torch.cat(self.buffer_autoencoder_n)
            x_ = self.autoencoder(x)
            error = mse(x, x_)
        self.threshold_autoencoder = self.alpha*error
            
        self.buffer_autoencoder_f = []
        self.buffer_autoencoder_n = []
        
    def update_predictor(self):

        self.predictor.update(self.buffer_predictor_n, self.buffer_predictor_f)

        with torch.no_grad():
            z = torch.cat([zy[0] for zy in self.buffer_predictor_n])
            y = [zy[1][0] for zy in self.buffer_predictor_n]
            y_pred = self.predictor(z)
            error = minMSELoss(y_pred, y)
        self.threshold_predictor = self.alpha*error

        self.buffer_predictor_f = []
        self.buffer_predictor_n = []

    def save_alexnet(self, path):
        d1 = self.autoencoder.encoder.state_dict()
        d1 = {'features.' + key:  d1[key] for key in d1}
        d2 = self.predictor.model.state_dict()
        d = {**d1, **d2}

        torch.save(d, path)
        print("Saved model to {}".format(path))
