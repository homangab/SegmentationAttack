import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import math
from PIL import Image
import torchvision.models as models
import sys
import random
import time
import pdb
#import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import copy
from torch.autograd.gradcheck import zero_gradients




__all__ = ['DRN', 'drn26', 'drn42', 'drn58']


webroot = 'https://tigress-web.princeton.edu/~fy/drn/models/'

model_urls = {
    'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
    'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
    'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
    'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
    'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
    'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False, out_middle=False, pool_size=28, arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                                   padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(
                BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(
                BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3,
                          bias=False),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(
                channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(
                channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4], dilation=2,
                                       new_level=False)
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], dilation=4,
                             new_level=False)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else \
                self._make_layer(BasicBlock, channels[6], layers[6], dilation=2,
                                 new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else \
                self._make_layer(BasicBlock, channels[7], layers[7], dilation=1,
                                 new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation=1)

        if num_classes > 0:
            self.avgpool = nn.AvgPool2d(pool_size)
            self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()

        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        x = self.layer4(x)
        y.append(x)

        x = self.layer5(x)
        y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)

        if self.out_map:
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

        if self.out_middle:
            return x, y
        else:
            return x


def drn_c_26(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-26']))
    return model


def drn_c_42(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-42']))
    return model


def drn_c_58(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-58']))
    return model


def drn_d_22(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-22']))
    return model


def drn_d_38(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-38']))
    return model


def drn_d_54(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-54']))
    return model


def drn_d_105(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-105']))
    return model





def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    if p ==np.inf:
            v = torch.clamp(v,-xi,xi)
    else:
        v = v * min(1, xi/(torch.norm(v,p)+0.00001))
    return v

def data_input_init(xi):
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    tf = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])
    
    v = (torch.rand(1,3,224,224).cuda()-0.5)*2*xi
    return (mean,std,tf,v)





def fool_perturb(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")


    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = image

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_variables=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_variables=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        r_i =  pert * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x[None, :])
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image    



def universal_perturbation(data_list, model, xi=10, delta=0.2, max_iter_uni = 10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10,init_batch_size = 1,t_p = 0.2):
    
    start_t = time.time()
    mean, std,tf,_ = data_input_init(xi)
    v = torch.autograd.Variable(torch.zeros(init_batch_size,3,224,224).cuda(),requires_grad=True)
    
    
    num_images =  len(data_list)
    # This is the batch size to be used for testing
    batch_size = init_batch_size
    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

    fooling_rate = 0.0
    itr = 0
    
    while fooling_rate < 1-delta and itr < max_iter_uni:
        batch_size = init_batch_size
        
        # Here the perturbations are computed for the segmentation
        random.shuffle(data_list)

        for k in range(0, num_batches):
            cur_img = torch.zeros(batch_size,3,224,224)
            data_inp = data_list[k*batch_size:min((k+1)*batch_size,len(data_list))]
            for i,name in enumerate(data_inp):
                im_orig = Image.open(name)
                cur_img[i] = tf(im_orig)
            #The current image for which uap is calculated    
            cur_img = torch.autograd.Variable(cur_img).cuda()
            batch_size = cur_img.size(0)
            #The ground truth labels in training dataset
            true_labels = np.argmax(model(cur_img).cpu().data.numpy(),1).astype(int)
            #The modified labels after perturbation
            pert_labels = np.argmax(model(cur_img+torch.stack((v[0],)*batch_size,0)).cpu().data.numpy(),1).astype(int)

            cor_stat = np.sum(true_labels==pert_labels)
            
            if (cor_stat/float(batch_size)) > 0:
                dr, iter, _, _, _ = fool_perturb((cur_img+torch.stack((v[0],)*batch_size,0)).data[0], model,num_classes= num_classes,
                                             overshoot= overshoot,max_iter= max_iter_df)
                        
                if iter < max_iter_df-1:
                    v.data = v.data + torch.from_numpy(dr).cuda()
                    # v.data = v.data + dr
                    # Project on l_p ball
                    v.data = proj_lp(v.data, xi, p)
                    
            if(k%10 ==0):
                print('>> k = ', k, ', pass #', itr)
                print('time for this',time.time()-start_t)
                print('Norm of v',torch.norm(v))
        batch_size = 100
        # To check the fooling rate achieved after the current iteration
        fooling_rate,model = get_fooling_rate(data_list,batch_size,v,model)
        itr = itr + 1
    
    return v

def get_fooling_rate(data_list,batch_size,v,model):
    
    # Perturb the dataset with computed perturbation
    tf = data_input_init(0)[2]
    #Datalist is the list of image names
    num_images = len(data_list)
    
    est_labels_pert = np.zeros((num_images))

    est_labels_orig = np.zeros((num_images))
    
    #This is the batch size used for testing 
    batch_size = 100
    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))
    # Compute the estimated labels in batches
    for ii in range(0, num_batches):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)
        #Intializing the set of images
        dataset = torch.zeros(M-m,3,224,224)
        dataset_perturbed =torch.zeros(M-m,3,224,224)
        #The main loop for computing fooling rate
        for iter,name in enumerate(data_list[m:M]):
            im_orig = Image.open(name)
            if (im_orig.mode == 'RGB'):
                dataset[iter] =  tf(im_orig)
                dataset_perturbed[iter] = tf(im_orig).cuda()+ v[0].data
            else:
                im_orig = torch.squeeze(torch.stack((tf(im_orig),)*3,0),1)
                dataset[iter] =  im_orig
                dataset_perturbed[iter] = im_orig.cuda()+ v[0].data
        dataset_var = torch.autograd.Variable(dataset,volatile = True).cuda()
        dataset_perturbed_var = torch.autograd.Variable(dataset_perturbed,volatile = True).cuda()

        est_labels_orig[m:M] = np.argmax(model(dataset_var).data.cpu().numpy(), axis=1).flatten()
        est_labels_pert[m:M] = np.argmax(model(dataset_perturbed_var).data.cpu().numpy(), axis=1).flatten()
        if ii%10 ==0:
            print(ii,'batches done.')

    # Finally, compute the fooling rate
    fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
    print('FOOLING RATE = ', fooling_rate)
    for param in model.parameters():
        param.volatile = False
        param.requires_grad = False
    
    return fooling_rate,model

   
def get_model(model):
    if model == 'vgg16':
        net = models.vgg16(pretrained=True)
    elif model =='resnet18':
        net = models.resnet18(pretrained=True)
    elif model =='drn_c_26':
        drn_c_26(pretrained=True)
    elif model =='drn_c_42':
        drn_c_42(pretrained=True)
    elif model =='drn_c_58':
        drn_c_58(pretrained=True)
    elif model =='drn_c_22':
        drn_c_22(pretrained=True)
    elif model =='drn_c_38':
        drn_c_38(pretrained=True)
    elif model =='drn_c_54':
        drn_c_54(pretrained=True)
    elif model =='drn_c_105':
        drn_c_105(pretrained=True)    
                            

    
    for params in net.parameters():
        requires_grad = False
    net.eval()
    net.cuda()
    return net
