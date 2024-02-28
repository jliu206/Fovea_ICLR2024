import torch
import torch.nn as nn
from utility import *
from ConvLSTM import *
import cv2
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from GTSR_classifier2 import Net
from torchvision.models.resnet import *
#device = torch.device('cuda:1')

class FoveaModel(nn.Module):
    def __init__(self,T,A,B,N):
        super(FoveaModel,self).__init__()
        self.T = T
        # self.batch_size = batch_size
        self.A = A
        self.B = B
        #self.z_size = z_size
        self.N = N
        #self.n = 28
        self.in_chan = 3

        #self.encoder = nn.LSTMCell(28*28, enc_size)
        self.encoder_l1 = ConvLSTMCell(input_dim=self.in_chan,
                                               hidden_dim=8,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_l2 = ConvLSTMCell(input_dim=8,
                                               hidden_dim=16,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_l3 = ConvLSTMCell(input_dim=16,
                                       hidden_dim=3,
                                       kernel_size=(3, 3),
                                       bias=True)
        self.CNN = nn.Conv2d(in_channels=3,
                                     out_channels=3,
                                     kernel_size=(3, 3),#10
                                     padding='same')

        self.linear1 = nn.Linear(12544, 100)
        self.linear2 = nn.Linear(100, 12544)
        self.decoder_l1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.decoder_l2 = nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=4, stride=2, padding=1)


    def forward(self,x, target, state_vector, samples = None):

        self.batch_size= x.size(0)
        # initialize hidden states
        h_enc_prev, enc_state, h_enc_prev2, enc_state2, h_enc_prev3, enc_state3 = state_vector
        loss = 0
        correct = [0] * self.T
        #x_original = x.clone()
        #First ts, with mask to be fixed
        mask, hr_mask = self.hard_mask2(samples)
        r_t0 = x * mask.to(torch.float32)
        hr = x * hr_mask.to(torch.float32)

        h_enc, enc_state = self.encoder_l1(r_t0, (h_enc_prev, enc_state))
        h_enc2, enc_state2 = self.encoder_l2(h_enc, (h_enc_prev2, enc_state2))
        h_enc3, enc_state3 = self.encoder_l3(h_enc2, (h_enc_prev3, enc_state3))
        hidden_final = self.CNN(h_enc3)

        reconstruction = hidden_final
        fovea = r_t0
        hidden = h_enc3


        return [h_enc, enc_state, h_enc2, enc_state2, h_enc3, enc_state3],\
               reconstruction, fovea, hr, hidden

    def hard_mask2(self, sample):
        """ Generate masked images w.r.t policy learned by the agent.
        """
        #input_full = input_org.clone()
        #sampled_img = torch.zeros([input_org.shape[0], input_org.shape[1], input_org.shape[2], input_org.shape[3]])
        patch_size = 32
        if sample == None:
            sample = torch.ones((self.batch_size,1))*5

        mask = torch.cuda.FloatTensor(self.batch_size, self.N, self.N).uniform_() > 0.95
        #mask = torch.zeros(self.batch_size, self.N, self.N).to(device)
        hr_img_mask = torch.cuda.FloatTensor(self.batch_size, self.N, self.N).uniform_() > 1
        for k in range(self.batch_size):
            x = sample[k]//7
            y = sample[k] % 7
            x = x.int()
            y = y.int()
            mask[k, x * patch_size: x * patch_size + patch_size,
                y * patch_size: y * patch_size + patch_size] = 1
            hr_img_mask[k, x * patch_size: x * patch_size + patch_size,
                y * patch_size: y * patch_size + patch_size] = 1
        mask = torch.unsqueeze(mask, 1)
        mask = mask.repeat(1, 3, 1, 1)
        hr_img_mask = torch.unsqueeze(hr_img_mask, 1)
        hr_img_mask = hr_img_mask.repeat(1, 3, 1, 1)
        return mask, hr_img_mask


    def generate(self,x, target):
        self.forward(x, target)
        imgs = []
        fovea_list = []
        hiddens = []
        for img in self.reconstruction:
            imgs.append(img.cpu().data.numpy())
        for fovea in self.foveas:
            fovea_list.append(fovea.cpu().data.numpy())
        for hid in self.hiddens:
            hiddens.append(hid.cpu().data.numpy())
        return imgs, fovea_list, hiddens

class Recurrent_Foveal_cell(nn.Module):
    def __init__(self,T,A,B,N):
        super(Recurrent_Foveal_cell, self).__init__()
        self.T = T
        self.N = N
        self.A = A
        self.B = B
        self.Foveal = FoveaModel(self.T, self.A, self.B, self.N)

        self.Foveal.load_state_dict(torch.load("save_imagenet_98masked_16fov_ssim_mse/weights_final.tar", map_location={'cuda:0':'cuda:2'}))
        self.policy = ResNet(BasicBlock, [2,2,2,2], num_classes=49)
        self.critic = ResNet(BasicBlock, [2,2,2,2], num_classes=1)
        self.classification = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
       # self.classification = self.classification.to(device)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        for param in self.Foveal.parameters():
            param.requires_grad = False
        for param in self.classification.parameters():
            param.requires_grad = False

    def forward(self, x, target, state_vector, samples, history_mask, last, training = True):
        self.batch_size = x.size(0)
        if len(state_vector) == 0:
            h_enc, enc_state = self.Foveal.encoder_l1.init_hidden(batch_size=self.batch_size, image_size=(self.B, self.A))
            h_enc2, enc_state2 = self.Foveal.encoder_l2.init_hidden(batch_size=self.batch_size, image_size=(self.B, self.A))
            h_enc3, enc_state3 = self.Foveal.encoder_l3.init_hidden(batch_size=self.batch_size, image_size=(self.B, self.A))
            state_vector = [h_enc, enc_state, h_enc2, enc_state2, h_enc3, enc_state3]


        state_vector, reconstruct, fovea, hr, hid\
            = self.Foveal(x, target, state_vector, samples)
        lr_map = torch.nn.functional.interpolate(reconstruct, (28, 28))
        value = self.critic(lr_map)

        preds = self.classification(reconstruct)
        preds = self.softmax(preds)
        R1, matches = compute_reward2(preds, target)
        correct = torch.sum(matches)

        pred_prob = self.policy(lr_map)  # should it be r_t?
        pred_prob = self.softmax(pred_prob+ torch.log(history_mask))
        log_probs, samples = torch.max(pred_prob, dim=1)
        if training:
            samples, log_probs, norm_prob = policy_sample(pred_prob)
        onehot_vector = one_hot(samples,num_classes = 49)
        history_mask -= onehot_vector
            #print(samples[:10])
        #print(pred_prob[:10])

        if last:
            return value

        return state_vector, R1, value, correct, reconstruct, fovea, hid, samples, log_probs