import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.utils.data as data_utils
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import random
import torchvision.models as models
import time
import math
from PIL import Image
import sys
sys.path.insert(0,'DeepFool/Python/')
from deepfool import deepfool



def univ_perturb(data_list, model,epsilon=0.2, max_iteration = np.inf, r=10/255.0, p=np.inf, max_iter_df=10,batch=50):
    
    
    mean, std,tf, v_i = data_input_init(r)
    v = torch.autograd.Variable(v_i.cuda(),requires_grad=True)

    
    
    
    global main_value
    main_value = [0]
    main_value[0] =torch.autograd.Variable(torch.zeros(1)).cuda()

    
    
    #Setting the current batch size to the initial batch size
    
    batch_size = batch
    
    opt = optim.Adam([v], lr = 0.1)

    # Count the number of images in the dataset
    image_tot =  len(data_list)

    batch_tot = np.int(np.ceil(np.float(image_tot) / np.float(batch_size)))
    

    #The initial fooling rate should be 0 as adversarial attack has not yet been applied
    fooling_rate = 0.0
    itr = 0

    model = set_norm(model)
    
    # epsilon is used as a control for the desired max fooling rate
    while fooling_rate < 1-epsilon and itr < max_iteration:
        
        # we shuffle the list of images to reduce any-undesired correlations between succeeding images
        random.shuffle(data_list)
        
        
        # Here the perturbations are computed for the segmentation
        for k in range(0, batch_tot):

            # M is the minimum of total images and (iteration index x size of one batch)
            # This is for dealing with the edge effects
            M = min((k+1)*batch_size,image_tot)

            #Matrix for the current image
            current = torch.zeros(batch_size,3,224,224)
            
            for j in range(k*batch_size,M):
                #the original image
                im_orig = Image.open(data_list[j])
                current[j%batch_size] = tf(im_orig)
            current = torch.autograd.Variable(current).cuda()
            
            opt.zero_grad()
            out = model(current+torch.stack((v[0],)*batch_size,0))
            loss = main_value[0]
            
            loss.backward()
            opt.step()
            main_value[0] = torch.autograd.Variable(torch.zeros(1)).cuda()
            v.data = project(v.data, r, p)
            if k%6 == 0 and k!=0:
                v.data = torch.div(v.data,2.0)
                
            
        batch_size = 100
        fooling_rate,model = fooling_rate(data_list,batch_size,v,model)
        itr+=1
    return v

def fooling_rate(data_list,batch_size,v,model):
    """
    :data_list: list of image names
    :batch_size: batch size to use for testing
    :model: the target network
    """
    # Using the perturbations calculated , we perturb the images
    tf = data_input_init(0)[2]
    image_tot = len(data_list)
    original_labels = np.zeros((image_tot))
    perturbed_labels = np.zeros((image_tot))

    batch_size = 100
    batch_tot = np.int(np.ceil(np.float(image_tot) / np.float(batch_size)))
    # Compute the estimated labels in batches
    for img in range(0, batch_tot):
        
        # M is the minimum of total images and (iteration index x size of one batch)
        # This is for dealing with the edge effects
        M = min((img+1)*batch_size, image_tot)

        m = (img * batch_size)

        #Initialize the data for both perturbed and un-perturbed cases
        data_1 = torch.zeros(M-m,3,224,224)
        data_1_perturbed =torch.zeros(M-m,3,224,224)

        for itr,name in enumerate(data_list[m:M]):
            original_img = Image.open(name)

            #If the mode of the images is RGB, we directly put them into the data set vector
            if (original_img.mode == 'RGB'):
                data_1[itr] =  tf(original_img)
                data_1_perturbed[itr] = tf(original_img).cuda()+ v[0].data
            #Else we first modify the images using torch.squeeze    
            else:
                original_img = torch.squeeze(torch.stack((tf(original_img),)*3,0),1)

                data_1_perturbed[itr] = original_img.cuda()+ v[0].data
                data_1[itr] =  original_img
                
        data_1_var = torch.autograd.Variable(data_1,volatile = True).cuda()
        data_1_perturbed_var = torch.autograd.Variable(data_1_perturbed,volatile = True).cuda()

        original_labels[m:M] = np.argmax(model(data_1_var).data.cpu().numpy(), axis=1).flatten()
        perturbed_labels[m:M] = np.argmax(model(data_1_perturbed_var).data.cpu().numpy(), axis=1).flatten()
        
    # This step finally computes the fooling rate
    fooling_rate = float(np.sum(perturbed_labels != original_labels) / float(image_tot))
    
    for param in model.parameters():
        param.volatile = False
        param.requires_grad = False
    
    return fooling_rate,model

def set_norm(model): 
    
    def get_norm(self,input,output):
        global main_value
        main_value[0] += -torch.log((torch.mean(torch.abs(output))))
    
    layers_to_opt = layers(model.__class__.__name__)
    print(layers_to_opt,'Layers')
    for name,layer in model.named_modules():
        if(name in layers_to_opt):
            print(name)
            layer.register_forward_hook(get_norm)
    return model  

def layers(model):
    if model =='VGG':
        layers_to_opt = [1,3,6,8,11,13,15,18,20,22,25,27,29]
        layers_to_opt = ['features.'+str(x) for x in layers_to_opt]
    elif 'ResNet' in model:
        layers_to_opt = ['conv1','layer1','layer2','layer3','layer4']
    return layers_to_opt
    
def get_model(model):
    if model == 'vgg16':
        net = models.vgg16(pretrained=True)
    elif model =='resnet18':
        net = models.resnet18(pretrained=True)
    
    for params in net.parameters():
        requires_grad = False
    net.eval()
    net.cuda()
    return net   


def data_input_init(r):
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    tf = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])
    
    v = (torch.rand(1,3,224,224).cuda()-0.5)*2*r
    return (mean,std,tf,v)


def project( c, r, p):

    # To project onto a sphere of radius r centered at c
    if p ==np.inf:
            c = torch.clamp(c,-r,r)
    else:
        c = c * min(1, r/(torch.norm(c,p)+0.00001))
    return c               