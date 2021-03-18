import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from time import sleep

class EWCModule(nn.Module):
    
    def __init__(self):
        super(EWCModule, self).__init__()
        
        self.gamma = 0.9

        
        self.prev_task = None
        self.est_fisher = None
        
        self.label = ""
        
        

    def estimate_fisher(self, data):

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()
    
        # Set self to evaluation mode
        mode = self.training
        self.eval()
    
        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for x in tqdm(data, desc="Computing Fisher informations for {}".format(self.label), leave=None):
    
            # run forward pass of self
            x = x.to(self.device)
            output = self(x)
    
            # -use predicted label to calculate loglikelihood:
            label = output.max(1)[1]
            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
    
            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()
    
            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2
    
            #sleep(0.01)

        # Normalize by sample size used for estimation
        #est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}
    
        # Store new values in the network
        existing_values = self.est_fisher
        self.prev_task = {}
        self.est_fisher = {}

        
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.prev_task[n] = p.detach().clone()
                # -precision (approximated by diagonal Fisher Information matrix)
                if existing_values is not None:
                    est_fisher_info[n] += self.gamma * existing_values[n]
                self.est_fisher[n] = est_fisher_info[n]
    
    
        # Set self back to its initial mode
        self.train(mode=mode)
        
    
    
    def ewc_loss(self):
        if self.est_fisher is not None:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                    n = n.replace('.', '__')
                    mean = self.prev_task[n]
                    fisher = self.est_fisher[n]
                    # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                    fisher = self.gamma*fisher
                    # Calculate EWC-loss
                    losses.append((fisher * (p-mean)**2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1./2)*sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self.device)



                
            