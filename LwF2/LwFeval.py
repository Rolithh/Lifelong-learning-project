# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:27:19 2021

@author: Laetitia Haye
"""

import torch
from data_loader_light2 import ContinualJacquardLoader
import numpy as np
from LwFmodel import model_n_tasks

n = 5
data_path = "../Jacquard_light2/"
grasp_norm = np.array([1024,1024,180,1024,1024])

model_path = "model{}".format(n-1)

#model = torch.load(model_path)
#model = torch.jit.load(model_path)

model = model_n_tasks(5).cuda()
model.load_state_dict(torch.load(model_path))
#model.eval()


dl = ContinualJacquardLoader(data_path)

for task in range(n):
    data = []
    datal = dl.get_data('test', task)
    
    for x, y in datal:
        
        y_all = model(x.cuda())
        
        #take the predictions from the task-specific neurones matching the task
        y = y_all.narrow(1, 5*task+1, 5)
        
        data.append(y)

    data = torch.cat(data)
    np.save('results{}_{}'.format(n,task), data.cpu().detach().numpy()*grasp_norm)