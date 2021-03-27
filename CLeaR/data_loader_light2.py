import numpy as np
import os
import glob

import time
import torch


grasp_norm = np.array([1024,1024,180,1024,1024])



def collate_fn(samples):
    xbatch = []
    ybatch = []
    
    for x, y in samples:
        if x is not None:
            xbatch.append(x.unsqueeze(0))
            ybatch.append(y)
        
    return torch.cat(xbatch), ybatch



class ContinualJacquardLoader:
    """Data loader for continual learning with Jacquard grasping dataset"""
    
    def __init__(self, data_path, batch_size=16, shuffle=True):
        """
        data_path : path of the directory containing the data
        task_n : number of tasks
        img_size : size of the images returned by the dataloader
        batch_size : batch size
        shuffle : if True, data is shuffled for each task
        """
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.tasks = [os.path.join(data_path, d) for d in os.listdir(data_path)]
        

        
        
        
    def get_data(self, phase, index):
        """Return a dataloader for the desired task
        phase : either 'train' or 'test'
        index : task index"""

        dataset = JacquardSubDataset(self.tasks[index], phase)
        dl = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=collate_fn)
        
        return dl
    
    def __getitem__(self, index):
        return self.get_data('train', index)
    
    
    def __len__(self):
        return len(self.tasks)
    
    
    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter < len(self.tasks):
           self.iter += 1
           return self.get_data('train', self.iter -1)
        else:
           raise StopIteration
        
        
        


class JacquardSubDataset(torch.utils.data.Dataset):
    """Sub dataset class for Jacquard grasping dataset"""
    
    def __init__(self, path, phase):
        """
        object_paths : list of directories
        phase : either 'train' or 'test'
        img_size : size of the images returned by the dataloader
        """
        
        graspf = [file for file in glob.glob(os.path.join(path, '*_grasps.npz')) if phase in file]
        
        self.inputs = []
        self.grasps = []
        
        for file in graspf:
            
            imgfile = file.replace('_grasps.npz', '.npz')
            self.inputs.append(np.load(imgfile)['arr_0'])
            
            gs = np.load(file)
            for k in range(len(gs)):
                self.grasps.append(gs['arr_' + str(k)]/grasp_norm)
            
        self.inputs = np.concatenate(self.inputs, axis=0)
        
        

    
    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
    
    def __getitem__(self, index):
        try:

            x = self.numpy_to_torch(self.inputs[index])
        
            y = self.numpy_to_torch(self.grasps[index])
            return x, y
        except:
            print('Error loading file')
            return None, None
    
    def __len__(self):
        return len(self.grasps)
    
    

    
