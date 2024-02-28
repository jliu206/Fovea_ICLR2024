import torch.optim as optim
from torchvision import datasets,transforms
import torch.utils
from RGBReco_A2C_R_imagenet2 import *
from get_meanstd import get_mean_and_std
#from Reco import  FoveaModel
from config_imgnet2 import *
#from utility import *
#from Mnist_classifier import *
import torch.nn.utils
import matplotlib.pyplot as plt

def create_peri(bs, ts):

    one_patch = torch.zeros(bs,7,7)
    if ts == 0:
        one_patch[:,1,1] = 1
    elif ts ==1:
        one_patch[:, 1, 5] = 1
    elif ts == 2:
        one_patch[:, 3, 3] = 1
    elif ts == 3:
        one_patch[:, 5, 1] = 1
    elif ts == 4:
        one_patch[:, 5,5] = 1
    all_patches = one_patch.repeat(1,32,32)

    return all_patches


if __name__ == '__main__':
    #train()
    #print('finishtraining')
    create_peri()