import torch.autograd as autograd
import torch
from config import *
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Multinomial, Bernoulli, Categorical
from torch.nn.functional import one_hot
#device = torch.device('cuda:0')

def Variable(data, *args, **kwargs):
    if USE_CUDA:
        data = data.cuda()
    return autograd.Variable(data,*args, **kwargs)

def unit_prefix(x, n=1):
    for i in range(n): x = x.unsqueeze(0)
    return x

def align(x, y, start_dim=0):
    xd, yd = x.dim(), y.dim()
    if xd > yd: y = unit_prefix(y, xd - yd)
    elif yd > xd: x = unit_prefix(x, yd - xd)

    xs, ys = list(x.size()), list(y.size())
    nd = len(ys)
    for i in range(start_dim, nd):
        td = nd-i-1
        if   ys[td]==1: ys[td] = xs[td]
        elif xs[td]==1: xs[td] = ys[td]
    return x.expand(*xs), y.expand(*ys)

def matmul(X,Y):
    results = []
    for i in range(X.size(0)):
        result = torch.mm(X[i],Y[i])
        results.append(result.unsqueeze(0))
    return torch.cat(results)

def xrecons_grid(x,B,A):
	"""
	plots canvas for single time step
	X is x_recons, (batch_size x img_size)
	assumes features = BxA images
	batch is assumed to be a square number
	"""
	padsize=1
	padval=.5
	ph=B+2*padsize
	pw=A+2*padsize
	batch_size=x.shape[0]
	N=int(np.sqrt(batch_size))

	x = np.transpose(x,(0,2,3,1)).reshape((N, N, B, A, 3))#delete 3 for i channel image
	img=np.ones((N*ph,N*pw, 3))*padval
	for i in range(N):
		for j in range(N):
			startr=i*ph+padsize
			endr=startr+B
			startc=j*pw+padsize
			endc=startc+A
			img[startr:endr,startc:endc]=x[i,j,:,:,:]
	return img

def save_image(x,count=0):

    for t in range(T):
        img = xrecons_grid(x[t][:1,:],B,A)
        plt.imshow(img)
        #plt.show()
        #plt.matshow(img, cmap=plt.cm.gray)
        imgname = 'save_imagenet_98_hybrid4/reco_count_%d_%s_%d.png' % (count,'test', t)  # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
        plt.savefig(imgname)
        print(imgname)

def save_image2(x_hat,count=0):

    for t in range(T):
        img = xrecons_grid(x_hat[t][:1,:],B,A)
        # img = torch.from_numpy(x_hat[t][:1,:])
        # img = torch.nn.functional.interpolate(img, (28, 28)).permute((0,2,3,1))
        # img = img[0]
        plt.imshow(img)
        #plt.show()
        #plt.matshow(img, cmap=plt.cm.gray)
        imgname = 'save_imagenet_98_hybrid4/fov_count_%d_%s_%d.png' % (count,'test', t)  # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
        plt.savefig(imgname)
        print(imgname)

def save_image3(x,count=0):
    ori_img = xrecons_grid(x[:1, :], B, A)
    plt.imshow(ori_img)
    #plt.show()
    imgname = 'save_imagenet_98_hybrid4/ori_count_%d_%s.png' % (count, 'test')
    plt.savefig(imgname)
    print(imgname)

def policy_sample(prob):
    distrib = Categorical(prob)
    samples = distrib.sample()
    norm_prob = distrib.probs
    norm_log = distrib.logits
    log_prob = distrib.log_prob(samples)

    return samples, log_prob, norm_prob

def compute_reward(preds, targets):
        #patch_use = torch.ones(preds.size(0))
        sparse_reward = torch.ones(preds.size(0))
        pred_value, pred_idx = preds.max(1)
        match = (pred_idx == targets).data
        #pred2 = preds.argmax(dim=1, keepdim=True)
        reward = sparse_reward
        reward[~match] = -1
        reward = reward.unsqueeze(1)

        return reward, match.float()


def compute_reward2(preds, targets):
    # patch_use = torch.ones(preds.size(0))
    sparse_reward = torch.ones(preds.size(0))
    pred_value, pred_idx = preds.max(1)
    match = (pred_idx == targets).data
    # pred2 = preds.argmax(dim=1, keepdim=True)
    reward = sparse_reward
    reward[~match] = -1
    reward = reward.unsqueeze(1)
    reward = reward.to(torch.device('cuda:0'))
    pred_value = pred_value.unsqueeze(1)
    final_reward = reward * pred_value
    return final_reward, match.float()


def preprocess_dataset(dataset,class_pick_list):
    new_targets = []
    new_imgs = []
    new_samples = []
    for i in range(len(dataset)):
        if dataset.targets[i] in [0]:
            new_targets.append(dataset.targets[i])
            new_imgs.append(dataset.imgs[i])
            new_samples.append((dataset.samples[i]))
    dataset.targets = new_targets
    dataset.imgs = new_imgs
    dataset.samples = new_samples
    return dataset