from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools

class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)
        
def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    return img_


def write_helper(x, results, ann_names):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    ann_file = ann_names[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])

    with open(ann_file, 'a') as out_file:
        out_file.write('%s %s %s %s %s %s\n' % (
                        label,
                        x[-2].item(),
                        c1[0].item(), c1[1].item(),
                        c2[0].item(), c2[0].item()))

    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img


def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument('--names', dest = 'namesfile', help = 
                        'Names file',
                        default = 'data/coco.names', type = str)
    parser.add_argument('--num-of-classes', dest = 'num_of_classes', help = 
                        'Number of classes',
                        default=80, type=int)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()

if __name__ ==  '__main__':
    args = arg_parse()
    
    scales = args.scales
    
    
#        scales = [int(x) for x in scales.split(',')]
#        
#        
#        
#        args.reso = int(args.reso)
#        
#        num_boxes = [args.reso//32, args.reso//16, args.reso//8]    
#        scale_indices = [3*(x**2) for x in num_boxes]
#        scale_indices = list(itertools.accumulate(scale_indices, lambda x,y : x+y))
#    
#        
#        li = []
#        i = 0
#        for scale in scale_indices:        
#            li.extend(list(range(i, scale))) 
#            i = scale
#        
#        scale_indices = li

    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = args.num_of_classes
    classes = load_classes(args.namesfile)

    #Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    #Set the model in evaluation mode
    model.eval()

    #Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    leftover = 0

    if (len(imlist) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        imlist_batches = [imlist[i*batch_size : min((i +  1)*batch_size,
                            len(imlist))]  for i in range(num_batches)]
    else:
        imlist_batches = [[x] for x in imlist]

    write = False
    # model(get_test_input(inp_dim, CUDA), CUDA)

    start_det_loop = time.time()

    for batch_i, imlist_batch in enumerate(imlist_batches):

        batch = list(map(prep_image, imlist_batch, [inp_dim for x in range(len(imlist_batch))]))
        im_batch = [x[0] for x in batch]
        orig_ims = [x[1] for x in batch]
        im_dim_list = [x[2] for x in batch]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

        if CUDA:
            im_dim_list = im_dim_list.cuda()

        batch = torch.cat(im_batch)

        start = time.time()
        if CUDA:
            batch = batch.cuda()


        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        #flatten the prediction vector 
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

#        prediction = prediction[:,scale_indices]


        #get the boxes with object confidence > threshold
        #Convert the cordinates to absolute coordinates
        #perform NMS on these boxes, and save the results 
        #I could have done NMS and saving seperately to have a better abstraction
        #But both these operations require looping, hence 
        #clubbing these ops in one loop instead of two. 
        #loops are slower than vectorised operations. 

        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)

        if type(prediction) == int:
            continue

        end = time.time()

        output = prediction

#        for im_num, image in enumerate(imlist_batch):
#            im_id = im_num
#            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
#            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
#            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
#            print("----------------------------------------------------------")

        if CUDA:
            torch.cuda.synchronize()

        im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

        scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)

        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])


        colors = pkl.load(open("pallete", "rb"))

        det_names = list(map(lambda x: os.path.join(args.det, os.path.split(x)[-1][:-3]+'txt'), imlist_batch))

        list(map(lambda x: write_helper(x, orig_ims, det_names), output))

        det_names = pd.Series(imlist_batch).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))

        list(map(cv2.imwrite, det_names, orig_ims))

        print("{0:d}-th batch finished in {1:6.3f} seconds".format(batch_i, end - start))
        print("----------------------------------------------------------")

    end = time.time()

    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", end - start_det_loop))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - start_det_loop)/len(imlist)))
    print("----------------------------------------------------------")


    torch.cuda.empty_cache()

