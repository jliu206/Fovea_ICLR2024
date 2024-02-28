import torch.optim as optim
from torchvision import datasets,transforms
import torch.utils
from imagenet_A2C_Modules import *
#from Reco import  FoveaModel
from config_imgnet import *
from utility2 import Variable,save_image,xrecons_grid,save_image2
#from Mnist_classifier import *
import torch.nn.utils
import matplotlib.pyplot as plt
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#torch.set_default_tensor_type('torch.FloatTensor')
from tqdm import *
from PIL import Image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_data = datasets.ImageFolder(
    '/data/jiayang/imagenet/train',
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        #transforms.Resize([112, 112]),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
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
    batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=64, shuffle=True)

model = Recurrent_Foveal_cell(T,A,B,N)
optimizer = optim.Adam(list(model.policy.parameters()) + list(model.critic.parameters()),lr=learning_rate,betas=(beta1,0.999))

if torch.cuda.is_available():
    device = torch.device('cuda:2')
    torch.cuda.set_device(device)
    model.cuda()
    print("use GPU")
else:
    device = torch.device('cpu')
    print("use CPU")

def compute_rewards(next_value, rewards, gamma=0.98):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R
        returns.insert(0, R)
    return returns

def train_one_epoch():
    avg_loss = 0
    total_correct =0
    base_total = 0
    for data, target in test_loader:

        bs = data.size()[0]
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        state_vector = []
        reconstruct_list = []
        fovea_list = []
        hid_list = []
        samples_list = []
        log_probs_list = []
        value_list = []
        R1_list = []
        correct_list = []
        samples = None
        base_samples = None
        history_mask = torch.ones((bs,49)).to(device)
        samples = torch.randint(0, 49, (bs, 1))
        onehot_vector = one_hot(samples, num_classes=49)
        onehot_vector = onehot_vector.squeeze(1).to(device)
        history_mask -= onehot_vector
        #base_samples = torch.randint(0, 16, (bs,))
        for i in range(T):
            state_vector, R1, value, correct, reconstruct, fovea, hid, samples, log_probs\
                = model(data, target, state_vector, samples, history_mask, last = False)
            reconstruct_list.append(reconstruct)
            fovea_list.append(fovea)
            hid_list.append(hid)
            samples_list.append(samples)
            log_probs_list.append(log_probs)
            value_list.append(value)
            R1_list.append(R1)
            correct_list.append(correct)

        next_value= \
            model(data, target, state_vector, samples, history_mask, last=True)
        reconstruct_list.append(reconstruct)
        total_correct += correct_list[-1]
        print('correct: {};'.format(correct_list[-1]))
        Q_list = compute_rewards(next_value, R1_list[1:])
        Q_tensor = torch.stack(Q_list).squeeze(2).transpose(1,0)
        # calculate loss
        log_probs_tensor = torch.stack(log_probs_list[1:]).transpose(1, 0)
        #R1_tensor = torch.stack(R1_list[1:]).transpose(1,0).squeeze(2).to(device)
        value_tensor = torch.stack(value_list[1:]).transpose(1,0).squeeze(2)
        Advantage = Q_tensor - value_tensor
        Advantage = Advantage.to(device)
        #print(Advantage[torch.where(Advantage!=0)])
        reinforce_loss = torch.sum(-log_probs_tensor * Advantage, dim=1)
        #reinforce_loss = torch.sum(-log_probs_tensor * 1/(Advantage+1e-10), dim=1)
        reinforce_loss = torch.mean(reinforce_loss, dim=0)
        critic_loss = Advantage.pow(2).mean()
        # if reinforce_loss <0:
        #     print('1')
        Total_loss = reinforce_loss + critic_loss
        Total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.policy.parameters())+list(model.critic.parameters()), clip)
        optimizer.step()
        print('loss: {};'.format(Total_loss))
        print("----------------------------")
    avg_loss += Total_loss.cpu().data.numpy()
    return total_correct
    #print('Epoch-{}; Count-{}; loss: {};'.format(epoch, count, avg_loss / 100))

def train():
    avg_loss = 0
    count = 0
    #model.policy.load_state_dict(torch.load("policy_models/policy_90_weights_5ts.tar"))
    model.classification.eval()
    model.Foveal.eval()
    k = []
    for epoch in tqdm(range(epoch_num)):
        a= train_one_epoch()
        k.append(a)
        print('toal_correct:',a)

    print(k)
    torch.save(model.policy.state_dict(), 'policy_models/imagenet_98_A2c_Softmax_5ts_49.tar')
    print('finish')

def test():
    avg_loss = 0
    model.policy.load_state_dict(torch.load("policy_models/imagenet_98_A2c_Softmax_4ts.tar"))
    model.classification.eval()
    model.Foveal.eval()
    model.policy.eval()
    total_correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            bs = data.size()[0]
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            state_vector = []
            base_state_vector = []
            reconstruct_list = []
            fovea_list = []
            hid_list = []
            samples_list = []
            log_probs_list = []
            samples = None
            #base_samples = None
            history_mask = torch.ones((bs,16)).to(device)
            history_mask[:,4] = 0
            # base_samples = torch.randint(0, 16, (bs,))
            for i in range(T):
                state_vector, R1, value, correct, reconstruct, fovea, hid, samples, log_probs \
                    = model(data, target, state_vector, samples,  history_mask,last=False, training = False)
                reconstruct_list.append(reconstruct)
                fovea_list.append(fovea)
                hid_list.append(hid)
                samples_list.append(samples)
                log_probs_list.append(log_probs)
                total_correct += correct
        print(total_correct)

if __name__ == '__main__':
    train()
    #print('finishtraining')
    #test()