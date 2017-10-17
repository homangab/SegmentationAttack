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