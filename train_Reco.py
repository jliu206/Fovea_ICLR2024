import torch.optim as optim
from torchvision import datasets,transforms
import torch.utils
from RGBReco_ae_ssim import FoveaModel
#from Reco import  FoveaModel
from config import *
from utility import Variable,save_image,xrecons_grid,save_image2, save_image3
#from Mnist_classifier import *
import torch.nn.utils
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim
import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.FloatTensor')
normalize = transforms.Normalize(mean = [0.3417, 0.3126, 0.3215], std = [0.2768, 0.2645, 0.2704])
normalize_test = transforms.Normalize(mean= [0.3372, 0.3095, 0.3207], std = [0.2722, 0.2627, 0.2723])
normalize_all = transforms.Normalize(mean= [0.3403, 0.3116, 0.3213], std = [0.2753, 0.2639, 0.2710])
train_data = datasets.GTSRB('data/', split="train", download=True,
                   transform=transforms.Compose([
                       transforms.Resize([112, 112]),
                       transforms.ToTensor(),
                       #normalize_all
                   ]))
test_data = datasets.GTSRB('data/', split='test', download=True,
                   transform=transforms.Compose([
                       transforms.Resize([224, 224]),
                       transforms.ToTensor(),
                       #normalize_all
                   ]))#GTSRB

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
train_data = datasets.ImageFolder(
    '/data/jiayang/imagenet/train',
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        #transforms.Resize([112, 112]),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #normalize,
    ]))

test_data = datasets.ImageFolder(
    '/data/jiayang/imagenet/val',
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.Resize([112, 112]),
        transforms.ToTensor(),
        #normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=16, shuffle=True)

model = FoveaModel(T,A,B,N)
#model.load_state_dict(torch.load('save_imagenet99/weights_final.tar'))
#model.eval()
optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=(beta1,0.999))
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    model.cuda()
    print("use GPU")
else:
    device = torch.device('cpu')
    print("use CPU")

def train():
    avg_loss = 0
    count = 0
    #model.load_state_dict(torch.load('save_imagenet_98masked_16fov_ssim_mse/weights_final.tar', map_location={'cuda:1':'cuda:0'}))
    for epoch in range(epoch_num):
        for data,t in train_loader:
            bs = data.size()[0]
            #If use Reco.py need this line
            #data = Variable(data).view(bs, -1)
            data = data.to(device)
            optimizer.zero_grad()
            loss,loss_list1, loss_list2 = model.loss(data)
            avg_loss += loss.cpu().data.numpy()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            count += 1
            if count % 100 == 0:
                print ('Epoch-{}; Count-{}; loss: {};'.format(epoch, count, avg_loss / 100))
                if count % 3000 == 0:
                    torch.save(model.state_dict(),'save_imagenet_98_hybrid4/weights_%d.tar'%(count))
                avg_loss = 0
    torch.save(model.state_dict(), 'save_imagenet_98_hybrid4/weights_final.tar')
    #generate_image()
    #generate_classification()

def generate_image():
    count = 0
    model.load_state_dict(torch.load('save_imagenet_98_hybrid4/weights_final.tar'))
    for data,target in test_loader:
            bs = data.size()[0]
            data = data.to(device)
            optimizer.zero_grad()
            ori, x, fovea, hidden = model.generate(data)

            save_image(x,count)
            save_image2(fovea,count)
            save_image3(ori,count)
            count += 1

if __name__ == '__main__':

    train()
    generate_image()#