import torch
from alexnet import load_model, minMSELoss
from data_loader_light2 import ContinualJacquardLoader
import numpy as np


n = 5
data_path = "../Jacquard_light2/"
model_path = "model{}".format(n-1)
grasp_norm = np.array([1024,1024,180,1024,1024])

model = load_model().cuda()
model.load_state_dict(torch.load(model_path))


dl = ContinualJacquardLoader(data_path)

for task in range(n):
    data = []
    datal = dl.get_data('test', task)
    
    for x, y in datal:
        
        y_ = model(x.cuda())
        
        data.append(y_)

    data = torch.cat(data)
    np.save('results{}_{}'.format(n,task), data.cpu().detach().numpy()*grasp_norm)