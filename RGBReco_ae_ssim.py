import torch
import torch.nn as nn
from utility import *
from ConvLSTM import *
import cv2
import torch.functional as F
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim
device = torch.device('cuda:1')
class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(SSIM_Loss, self).forward(img1, img2) )
class FoveaModel(nn.Module):
    def __init__(self,T,A,B,N):
        super(FoveaModel,self).__init__()
        self.T = T
        # self.batch_size = batch_size
        self.A = A
        self.B = B
        #self.z_size = z_size
        self.N = N
        self.original = [0]
        self.reconstruction = [0] * T
        self.foveas = [0] * T
        self.hiddens = [0] * T
        #self.n = 28
        self.in_chan = 3

        #self.encoder = nn.LSTMCell(28*28, enc_size)
        self.encoder_l1 = ConvLSTMCell(input_dim=self.in_chan,
                                               hidden_dim=8,
                                               kernel_size=(3,3),
                                               bias=True)
        self.pooling1 = nn.MaxPool2d(2)
        self.encoder_l2 = ConvLSTMCell(input_dim=8,
                                               hidden_dim=16,
                                               kernel_size=(3,3),
                                               bias=True)
        self.pooling2 = nn.MaxPool2d(2)
        self.encoder_l3 = ConvLSTMCell(input_dim=16,
                                       hidden_dim=3,
                                       kernel_size=(3,3),
                                       bias=True)
        self.linear1 = nn.Linear(12544,100)
        self.linear2 = nn.Linear(100,12544)
        self.decoder_l1 = nn.ConvTranspose2d(in_channels=16, out_channels= 8, kernel_size=4, stride=2, padding=1)
        self.decoder_l2 = nn.ConvTranspose2d(in_channels=8, out_channels= 3, kernel_size=4, stride=2, padding=1)
        self.CNN = nn.Conv2d(in_channels=3,
                                     out_channels=3,
                                     kernel_size=(3,3),#10
                                     padding='same')
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        #self.sample = torch.cuda.FloatTensor(self.N, self.N).uniform_() > 0.9

    def forward(self,x):
        self.batch_size= x.size(0)

        # initialize hidden states
        h_enc_prev, enc_state = self.encoder_l1.init_hidden(batch_size=self.batch_size, image_size=(self.B, self.A))
        h_enc_prev2, enc_state2 = self.encoder_l2.init_hidden(batch_size=self.batch_size, image_size=(self.B, self.A))
        h_enc_prev3, enc_state3 = self.encoder_l3.init_hidden(batch_size=self.batch_size, image_size=(self.B, self.A))
        #h_dec_prev, dec_state = self.decoder_l1.init_hidden(batch_size=self.batch_size, image_size=(self.B, self.A))
        all_mask = self.hard_mask2()
        #self.original = x
        # for t in range(self.T):
        #     r_t = x * all_mask[t].to(torch.float32)
        #     h_enc,enc_state = self.encoder_l1(r_t,(h_enc_prev,enc_state))
        #     h_enc_copy = h_enc.clone()
        #     h_enc_copy = self.pooling1(h_enc_copy)
        #
        #     h_enc2,enc_state2 = self.encoder_l2(h_enc_copy, (h_enc_prev2, enc_state2))
        #     h_enc2_copy = h_enc2.clone()
        #     h_enc2_copy = self.pooling2(h_enc2_copy)
        #
        #     h_enc2_copy = h_enc2_copy.view(h_enc2_copy.size(0), -1)
        #     h_1 = self.linear1(h_enc2_copy)
        #     h_2 = self.linear2(h_1)
        #     h_2 = h_2.reshape((h_2.size(0),16,28,28))
        #     h_dec1 = self.decoder_l1(h_2)
        #     h_dec2 = self.decoder_l2(h_dec1)
        #
        #     hidden_final = self.CNN(h_dec2)
        #     self.reconstruction[t] = self.sigmoid(hidden_final)
        #     self.foveas[t] = r_t
        #     self.hiddens[t] = h_1
        #     h_enc_prev = h_enc
        #     h_enc_prev2 = h_enc2


        #ae2
        self.original = x
        for t in range(self.T):
            r_t = x * all_mask[t].to(torch.float32)
            h_enc,enc_state = self.encoder_l1(r_t,(h_enc_prev,enc_state))
            h_enc2,enc_state2 = self.encoder_l2(h_enc, (h_enc_prev2, enc_state2))
            h_enc3, enc_state3 = self.encoder_l3(h_enc2, (h_enc_prev3, enc_state3))

            hidden_final = self.CNN(h_enc3)
            self.reconstruction[t] = hidden_final
            self.foveas[t] = r_t
            self.hiddens[t] = h_enc3
            h_enc_prev = h_enc
            h_enc_prev2 = h_enc2
            h_enc_prev3 = h_enc3

    def loss(self,x):
        self.forward(x)
        #criterion = nn.MSELoss()
        criterion = SSIM_Loss(data_range=1.0, size_average=True, channel=3)
        criterion2 = nn.MSELoss()
        loss = 0
        loss_list1 = []
        loss_list2 = []
        # #Just for last timestep's output
        # x_recons = self.sigmoid(self.reconstruction[-1])
        # #x_recons = self.tanh(self.relu(self.reconstruction[-1]))
        # #x_recons = self.reconstruction[-1]
        # Lx = criterion(x_recons,x)
        # loss = Lx    # 12
        #
        #Try to minimize all ts's output
        #the following three line is for denormalized input
        mean = torch.as_tensor([0.3403, 0.3116, 0.3213]).to(device)[None, :, None, None]
        std = torch.as_tensor([0.2753, 0.2639, 0.2710]).to(device)[None, :, None, None]

        mean = torch.as_tensor([0.485, 0.456, 0.406]).to(device)[None, :, None, None]
        std = torch.as_tensor([0.229, 0.224, 0.225]).to(device)[None, :, None, None]
        #ori_img = x * std + mean  # in [0, 1]

        #ori_img = ori_img.permute((0, 2, 3, 1))
        for t in range(len(self.reconstruction)):
            x_recons = self.reconstruction[t]
            a = criterion(x_recons,x)
            b = 100 * criterion2(x_recons, x)
            Lx = a + b
            loss += Lx
            loss_list1.append(a)
            loss_list2.append(b)
        #
        return loss, loss_list1, loss_list2

    def hard_mask2(self):
        """ Generate masked images w.r.t policy learned by the agent.
        """
        #input_full = input_org.clone()
        #sampled_img = torch.zeros([input_org.shape[0], input_org.shape[1], input_org.shape[2], input_org.shape[3]])
        all_mask = []
        patch_size = 56
        mask_patch = torch.ones((self.batch_size,patch_size,patch_size))

        rand_x = torch.randint(0,4,(self.batch_size,self.T))
        rand_y = torch.randint(0,4,(self.batch_size,self.T))

        # rand_x = torch.zeros((self.batch_size,5), dtype=torch.int64)
        # rand_x[:, 3] = 3
        # rand_x[:, 4] = 3
        #
        # rand_y = torch.tensor([0,1,3,0,3], dtype=torch.int64)
        # rand_y = torch.unsqueeze(rand_y,0)
        # rand_y = rand_y.repeat(self.batch_size,1)
        #
        # rand_x = torch.ones((self.batch_size, 5), dtype=torch.int64)
        # rand_x[:, 2] = 2
        # rand_x[:, 3] = 2
        # rand_x[:, 4] = 2
        # rand_y = torch.tensor([1,2,1,2,3], dtype=torch.int64)
        # rand_y = torch.unsqueeze(rand_y,0)
        # rand_y = rand_y.repeat(self.batch_size,1)

        # rand_x = torch.ones((self.batch_size,5), dtype=torch.int64)
        # rand_y = torch.ones((self.batch_size, 5), dtype=torch.int64)

        for i in range(rand_x.shape[1]):
            mask = torch.cuda.FloatTensor(self.batch_size, self.N, self.N).uniform_() > 0.98
            #mask = torch.zeros(self.batch_size, self.N, self.N)
            for k in range(self.batch_size):
                mask[k, rand_x[k, i] * patch_size: rand_x[k, i] * patch_size + patch_size,
                    rand_y[k, i] * patch_size: rand_y[k, i] * patch_size + patch_size] = 1
            mask = torch.unsqueeze(mask, 1)
            mask = mask.repeat(1, 3, 1, 1)
            all_mask.append(mask.cuda())
        return all_mask

    # correct
    def read_2(self,x):
        mask = self.hattn_mask(224)
        def filter_img(img,mask):
            mask = mask.to(torch.float32)
            return img * mask
        x_fovea = filter_img(x,mask)
        #return torch.cat((x,x_hat),1), x_hat
        return x_fovea

    def generate(self,x):
        self.forward(x)
        imgs = []
        fovea_list = []
        hiddens = []
        mean = torch.as_tensor([0.3403, 0.3116, 0.3213]).to(device)[None, :, None, None]
        std = torch.as_tensor([0.2753, 0.2639, 0.2710]).to(device)[None, :, None, None]
        #ori = self.original * std + mean
        ori = self.original.cpu().data.numpy()
        for img in self.reconstruction:
            imgs.append(img.cpu().data.numpy())
        for fovea in self.foveas:
            fovea_list.append(fovea.cpu().data.numpy())
        for hid in self.hiddens:
            hiddens.append(hid.cpu().data.numpy())

        return ori, imgs, fovea_list, hiddens

