import numpy as np

from skimage.transform import resize


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def preprocess_rgb(rgb, size):
    rgb = resize(rgb, (size, size), preserve_range=True).astype('float32')
    rgb /= 255.0
    rgb -= mean
    rgb /= std
    rgb = np.transpose(rgb, (2, 0, 1))
    
    return rgb


def preprocess_depth(depth, size):
    depth = depth.astype('float32')
    depth = resize(depth, (size, size), preserve_range=True)
    # TO DO
    return depth

def preprocess_grasps(grasps):
    # TO DO
    return grasps