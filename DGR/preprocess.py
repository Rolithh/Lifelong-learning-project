import numpy as np

from skimage.transform import resize


mean_rgb = np.array([0.485, 0.456, 0.406])
std_rgb = np.array([0.229, 0.224, 0.225])

mean_depth = 1.498
std_depth = 0.0478

def preprocess_rgb(rgb, size):
    rgb = resize(rgb, (size, size), preserve_range=True).astype('float32')
    rgb /= 255.0
    rgb -= mean_rgb
    rgb /= std_rgb
    rgb = np.transpose(rgb, (2, 0, 1))
    
    return rgb


def preprocess_depth(depth, size):
    depth = depth.astype('float32')
    depth = resize(depth, (size, size), preserve_range=True)
    depth -= mean_depth
    depth /= std_depth
    return depth

def preprocess_grasps(grasps):
    # TO DO
    return grasps