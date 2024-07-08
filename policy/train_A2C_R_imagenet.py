import torch.optim as optim
from torchvision import datasets,transforms
import torch.utils
from RGBReco_A2C_R_imagenet import *
from get_meanstd import get_mean_and_std
#from Reco import  FoveaModel
from config_imgnet import *
#from utility import *
#from Mnist_classifier import *
import torch.nn.utils
import matplotlib.pyplot as plt
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#torch.set_default_tensor_type('torch.FloatTensor')
from tqdm import *
from PIL import Image

def preprocess_dataset(dataset,class_pick_list):
    new_targets = []
    new_imgs = []
    new_samples = []
    for i in range(len(dataset)):
        if dataset.targets[i] in class_pick_list:
            new_targets.append(dataset.targets[i])
            new_imgs.append(dataset.imgs[i])
            new_samples.append((dataset.samples[i]))
    dataset.targets = new_targets
    dataset.imgs = new_imgs
    dataset.samples = new_samples
    return dataset


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_data = datasets.ImageFolder(
    '/data/jiayang/imagenet/train',
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        #transforms.Resize([112, 112]),
        # transforms.RandomHorizontalFlip(),
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
test_data = datasets.ImageFolder(
    '/home/jiayang/forvea/draw_pytorch/ICLR_appendix/fix_compare/',
    transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        #transforms.Resize([112, 112]),
        transforms.ToTensor(),
        #normalize,
    ]))
a = torch.randperm(1000)[:100]
random300_data = test_data
#random300_data = preprocess_dataset(test_data, a)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=1, shuffle=True)
test_loader2 = torch.utils.data.DataLoader(
    random300_data,
    batch_size=32, shuffle=True)
model = Recurrent_Foveal_cell(T,A,B,N)
optimizer = optim.Adam(list(model.policy.parameters()) + list(model.critic.parameters()),lr=learning_rate,betas=(beta1,0.999))

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    model.cuda()
    print("use GPU")
else:
    device = torch.device('cpu')
    print("use CPU")


def train_one_epoch():
    avg_loss = 0
    total_correct =0
    base_total = 0
    for data, target in train_loader:

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
        history_mask = torch.ones((bs,16)).to(device)
        samples = torch.randint(0, 16, (bs, 1))
        onehot_vector = one_hot(samples, num_classes=16)
        onehot_vector = onehot_vector.squeeze(1).to(device)
        history_mask -= onehot_vector
        #base_samples = torch.randint(0, 16, (bs,))
        for i in range(T):
            state_vector, R1, value, correct, reconstruct, fovea, hid, samples, log_probs, class_probs \
                = model(data, target, state_vector, samples, history_mask)
            reconstruct_list.append(reconstruct)
            fovea_list.append(fovea)
            hid_list.append(hid)
            samples_list.append(samples)
            log_probs_list.append(log_probs)
            value_list.append(value)
            R1_list.append(R1)
            correct_list.append(correct)
        print('correct: {};'.format(correct_list[-1]))
        # calculate loss
        log_probs_tensor = torch.stack(log_probs_list[:-1]).transpose(1, 0)
        R1_list[1] -= 0.1
        R1_list[2] -= 0.2
        R1_list[3] -= 0.3
        R1_list[4] -= 0.4
        R1_tensor = torch.stack(R1_list[1:]).transpose(1,0).squeeze(2).to(device)
        value_tensor = torch.stack(value_list[1:]).transpose(1,0).squeeze(2)
        Advantage = R1_tensor - value_tensor
        Advantage = Advantage.to(device)
        #print(Advantage[torch.where(Advantage!=0)])
        reinforce_loss = torch.sum(-log_probs_tensor * Advantage, dim=1)
        #reinforce_loss = torch.sum(-log_probs_tensor * 1/(Advantage+1e-10), dim=1)
        reinforce_loss = torch.mean(reinforce_loss, dim=0)
        critic_loss = Advantage.pow(2).mean()
        Total_loss = reinforce_loss + critic_loss
        Total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.policy.parameters())+list(model.critic.parameters()), clip)
        optimizer.step()
        print('loss: {};'.format(Total_loss))
        print("----------------------------")
        avg_loss += Total_loss.cpu().data.numpy()
        total_correct += correct_list[-1]
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
    torch.save(model.policy.state_dict(), 'policy_models/imagenet1000_98_policy_A2C_masked_nostd_weights_5ts_S+M.tar')
    torch.save(model.critic.state_dict(), 'policy_models/imagenet1000_98_critic_A2C_masked_nostd_weights_5ts_S+M.tar')
    print('finish')

def draw(img_list, aaa, name):
    fig = plt.figure()
    # fig2 = plt.figure()
    # setting values to rows and column variables
    rows = 1
    columns = T
    for t in range(T):
        img = np.transpose(img_list[t][0].cpu(), (1, 2, 0))
        # plt.imshow(img)
        fig.add_subplot(rows, columns, t + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title("timestep {}".format(t))
    imgname = '/home/jiayang/forvea/draw_pytorch/ICLR_appendix/fix_compare/%s_%d.png' % (name ,aaa)

    plt.savefig(imgname,bbox_inches = 'tight', dpi=300, pad_inches = 0.05)
    plt.show()

def test():
    avg_loss = 0
    model.policy.load_state_dict(torch.load("policy_models/imagenet1000_98_policy_A2C_masked_nostd_weights_5ts_S+M.tar"))
    model.critic.load_state_dict(torch.load("policy_models/imagenet1000_98_critic_A2C_masked_nostd_weights_5ts_S+M.tar"))
    model.classification.eval()
    model.Foveal.eval()
    model.policy.eval()
    model.critic.eval()
    total_correct = 0
    total = 0
    counter = 0
    aaa = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):

            bs = data.size()[0]
            data = data.to(device)
            #target += 500
            target = target.to(device)
            optimizer.zero_grad()
            state_vector = []
            base_state_vector = []
            reconstruct_list = []
            fovea_list = []
            hid_list = []
            samples_list = []
            log_probs_list = []
            R1_list = []
            class_probs_list = []
            class_list = []
            sum_results_preds = torch.zeros((bs, 1000)).to(device)
            samples = None
            #base_samples = None
            history_mask = torch.ones((bs,16)).to(device)
            samples = torch.randint(0, 16, (bs, 1))
            onehot_vector = one_hot(samples, num_classes=16)
            onehot_vector = onehot_vector.squeeze(1).to(device)
            history_mask -= onehot_vector

            # base_samples = torch.randint(0, 16, (bs,))
            accum_fovea_mask = torch.cuda.FloatTensor(bs, N, N).uniform_() > 1
            for i in range(T):
                #samples = samples_list[i]
                state_vector, R1, value, correct, reconstruct, fovea, hid, samples, log_probs, class_probs \
                    = model(data, target, state_vector, samples,  history_mask,accum_fovea_mask, i, training = False)
                prod_value, prob_idx = class_probs.max(1)

                if len(class_probs_list) != 0 and prod_value > 0.5 and class_probs_list[-1]>0.5 and prob_idx == class_list[-1]:
                    counter += (4-i)
                    reconstruct_list.append(reconstruct)
                    fovea_list.append(fovea)
                    break
                class_probs_list.append(prod_value)
                class_list.append(prob_idx)
                reconstruct_list.append(reconstruct)
                fovea_list.append(fovea)
            while len(reconstruct_list) != 5:
                reconstruct_list.append(torch.ones(1,3,224,224))
                fovea_list.append(torch.ones(1,3,224,224))
                #pass
            aaa += 3
            draw(reconstruct_list, aaa, name='recon')
            draw(fovea_list, aaa, name='fovea')
            total_correct += correct
            total += 1
            if total % 1000 == 0:
                print('correct: {}/{};'.format(total_correct, total))
                print('patches saved: {};'.format(counter))
        print(total_correct)

if __name__ == '__main__':
    #train()
    #print('finishtraining')
    test()
