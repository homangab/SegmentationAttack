#############################################################
import New_utils
import torch.backends.cudnn as cudnn
cudnn.enabled = False
##############################################################
from docopt import docopt
import time
import torch
import torchvision
import numpy as np
import torch.optim as optim

docstr = """Find Universal Adverserial Perturbations for Dilated Residual Networks.

Usage:
  uap_drn.py <model> <im_path> <im_list> [options]
  uap_drn.py (-h | --help)
  uap_drn.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --save_loc=<str>            Location for saving the UAP as FloatTensor[default: same_dir]
  --batch_size=<int>          batch_size for processing while forming UAP in gpu[default: 25]
  --gpu=<bool>                Which GPU to use[default: 3]
  --max_iter_uni=<int>        maximum epochs to train for[default: 10]   
  --xi=<float>                controls the l_p magnitude of the perturbation[default: 0.1866]
  --delta=<float>             controls the desired fooling rate[default: 0.2]
  --p=<float>                 norm to be used for the UAP[default: inf]
  --num_classes=<int>         For deepfool: num_classes (limits the number of classes to test against)[default: 10]
  --overshoot=<float>         For deepfool: used as a termination criterion to prevent vanishing updates[default: 0.02]
  --max_iter_df=<int>         For deepfool: maximum number of iterations for deepfool[default: 10]
  --t_p=<float>               For batch deepfool: truth perentage, for how many flipped labels in a batch atleast.[default: 0.2]
"""

if __name__ == '__main__':
    start_time = time.time() 
    args = docopt(docstr, version='v1.0')
    print (args['--gpu'])
    torch.cuda.set_device(int(args['--gpu']))
    
    #This is the model which will be attacked by universal adversarial perturbations
    net = New_utils.get_model(args['<model>'])
    
    #The path to folder containing images
    location_img = args['<im_path>']

    max_iter_uni=int(args['--max_iter_uni']) 

    delta=float(args['--delta']) 

    #The list of all image names
    img_list = args['<im_list>']

    xi=float(args['--xi'])
    
    if(args['--p'] == 'inf'):
        p = np.inf
    else:
        p=int(args['--p'])
    if(args['--save_loc'] == 'same_dir'):
        save_loc = '.'
    else:
        save_loc = args['--save_loc'] 
    num_classes=int(args['--num_classes'])
    overshoot=float(args['--overshoot'])
    max_iter_df=int(args['--max_iter_df'])
    t_p=float(args['--t_p'])
    
    file = open(img_list)
    img_names = []
    for f in file:
        img_names.append(f.split(' ')[0])
    img_names = [location_img +x for x in img_names]
    st = time.time()
    
    batch_size = 1
    uap = New_utils.universal_perturbation(img_names, net, xi=xi, delta=delta, max_iter_uni =max_iter_uni,
                                                      p=p, num_classes=num_classes, overshoot=overshoot, 
                                                      max_iter_df=max_iter_df,init_batch_size = batch_size,t_p = t_p)
            
    print('found uap.Total time: ' ,time.time()-st)
    uap = uap.data.cpu()
    torch.save(uap,save_loc+'uap_drn_'+args['<model>']+'.pth')
