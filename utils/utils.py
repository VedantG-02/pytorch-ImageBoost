import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math


# helper functions
# func to save model and optimizer 
def saveModel(model, optimizer, path):
    torch.save({
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
    }, path)


# load model 
def loadModel(path):
    return torch.load(path)


# Peak Signal-to-Noise Ratio metric func
def PSNR(sr, hr):
    '''
        Calculate PSNR value between sr image and hr image
    '''
    mse = nn.MSELoss()
    R = 1
    mse_value = mse(sr, hr).item()
    psnr = 10*math.log10((R**2)/mse_value)
    return psnr


# func that saves image to the given path 
def saveImage(lr, sr, hr, path, name_id):
    '''
        Convert lr, sr, hr to RGB image and then save
    '''
    transform_to_pil = transforms.ToPILImage()
    lr = transform_to_pil(lr)
    sr = transform_to_pil(sr)
    hr = transform_to_pil(hr)
    lr.save(f"{path}/{name_id:0>2}_original_LR.png")
    hr.save(f"{path}/{name_id:0>2}_original_HR.png")
    sr.save(f"{path}/{name_id:0>2}_SR.png")