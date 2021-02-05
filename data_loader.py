import numpy as np
import os
import glob
from imageio import imread

import torch

from preprocess import preprocess_rgb, preprocess_depth, preprocess_grasps


def collate_fn(samples):
    xbatch = []
    ybatch = []
    
    for x, y in samples:
        xbatch.append(x.unsqueeze(0))
        ybatch.append(y)
        
    return torch.cat(xbatch), ybatch



class ContinualJacquardLoader:
    """Data loader for continual learning with Jacquard grasping dataset"""
    
    def __init__(self, data_path, task_n, img_size=224, batch_size=16, shuffle=True):
        """
        data_path : path of the directory containing the data
        task_n : number of tasks
        img_size : size of the images returned by the dataloader
        batch_size : batch size
        shuffle : if True, data is shuffled for each task
        """
        
        self.task_n = task_n
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        graspf = glob.glob(os.path.join(data_path, '*', '*_grasps.txt'))
        
        self.object_paths = list(set([os.path.dirname(f) for f in graspf]))
        self.object_paths.sort()
        
        self.objects_per_task = len(self.object_paths) // task_n
        
        
    def get_data(self, phase, index):
        """Return a dataloader for the desired task
        phase : either 'train' or 'test'
        index : task index"""
        idx = index*self.objects_per_task
        paths = self.object_paths[idx:idx + self.objects_per_task]
        dataset = JacquardSubDataset(paths, phase, self.img_size)
        dl = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=collate_fn)
        
        return dl
    
    def __getitem__(self, index):
        return self.get_data('train', index)
    
    
    def __len__(self):
        return self.task_n
    
    
    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter < self.task_n:
           self.iter += 1
           return self.get_data('train', self.iter -1)
        else:
           raise StopIteration
        
        
        


class JacquardSubDataset(torch.utils.data.Dataset):
    """Sub dataset class for Jacquard grasping dataset"""
    
    def __init__(self, object_paths, phase, img_size=224):
        """
        object_paths : list of directories
        phase : either 'train' or 'test'
        img_size : size of the images returned by the dataloader
        """
        
        self.img_size = img_size
        
        self.files = []
        
        for path in object_paths:
            grasp_files = glob.glob(os.path.join(path, '*_grasps.txt'))
            grasp_files.sort()
            if phase == 'train':
                for f in grasp_files[:-1]:
                    self.files.append(f.replace('_grasps.txt', ''))
            elif phase == 'test':
                self.files.append(graps_files[-1].replace('_grasps.txt', ''))
        
    def get_rgb(self, idx):
        rgb = imread(self.files[idx] + '_RGB.png')
        return preprocess_rgb(rgb, self.img_size)
    
    def get_depth(self, idx):
        depth = imread(self.files[idx] + '_perfect_depth.tiff')
        return preprocess_depth(depth, self.img_size)
    
    def get_grasps(self, idx):
        path = self.files[idx] + '_grasps.txt'
        
        grasps = []
        with open(path, 'r') as file:
            for line in file:
                grasps.append([float(v) for v in line[:-1].split(';')])
                
        return preprocess_grasps(np.array(grasps))
        
    
    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
    
    def __getitem__(self, index):
        rgb = self.get_rgb(index)
        depth = self.get_depth(index)
        grasps = self.get_grasps(index)
        
        x = self.numpy_to_torch(
            np.concatenate((rgb, np.expand_dims(depth, 0)), axis=0)
        )
        
        y = self.numpy_to_torch(grasps)
        
        return x, y
    
    def __len__(self):
        return len(self.files)
    
    

    