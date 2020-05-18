# -*- coding: utf-8 -*

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import time
import argparse
import numpy as np
import json
import collections
import math

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import imdbfolder as imdbfolder
from spottune_models import *
# import models
import agent_net

from utils import *
from gumbel_softmax import *
import pickle
from make_small_dataset import *


use_multitune = True
use_air = True
run_small = False

parser = argparse.ArgumentParser(description='PyTorch SpotTune')
parser.add_argument('--nb_epochs', default=110, type=int, help='nb epochs')
#parser.add_argument('--nb_epochs', default=50, type=int, help='nb epochs')

parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate of net')
parser.add_argument('--lr_agent', default=0.01, type=float, help='initial learning rate of agent')

parser.add_argument('--datadir', default='./decathlon-1.0-data/', help='folder containing data folder')
parser.add_argument('--imdbdir', default='./decathlon-1.0-devkit/decathlon-1.0/annotations/', help='annotation folder')
parser.add_argument('--ckpdir', default='./cv/', help='folder saving checkpoint')

parser.add_argument('--seed', default=0, type=int, help='seed')

if use_multitune:
    parser.add_argument('--step1', default=20, type=int, help='nb epochs before first lr decrease')
    parser.add_argument('--step2', default=50, type=int, help='nb epochs before second lr decrease')
    parser.add_argument('--step3', default=80, type=int, help='nb epochs before third lr decrease')
else:
    parser.add_argument('--step1', default=40, type=int, help='nb epochs before first lr decrease')
    parser.add_argument('--step2', default=60, type=int, help='nb epochs before second lr decrease')
    parser.add_argument('--step3', default=80, type=int, help='nb epochs before third lr decrease')

args = parser.parse_args()

if use_air:
    weight_decays = [("aircraft", 0.0005)]
    datasets = [("aircraft", 0),]
else:
    weight_decays = [("cifar100", 0.0)]
    datasets = [("cifar100", 0)]    

datasets = collections.OrderedDict(datasets)
weight_decays = collections.OrderedDict(weight_decays)

with open(args.ckpdir + '/weight_decays.json', 'w') as fp:
    json.dump(weight_decays, fp)

def train(dataset, poch, train_loader, net, agent, net_optimizer, agent_optimizer, w0_dict):
    #Train the model
    net.train()
    agent.train()

    total_step = len(train_loader)
    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter()
    
    for i, task_batch in enumerate(train_loader):
        images = task_batch[0] 
        labels = task_batch[1]    
        
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        
        images, labels = Variable(images), Variable(labels)	   
        probs = agent(images)

        action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
        policy = action[:,:,1]
        
        outputs = net.forward(images, use_multitune, policy)
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(labels.data).cpu().sum()
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))

        loss = criterion(outputs, labels)
        tasks_losses.update(loss.item(), labels.size(0))

        if i % 50 == 0:
            print ("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
                .format(epoch+1, args.nb_epochs, i+1, total_step, tasks_losses.val, tasks_top1.val, tasks_top1.avg))
       
        #---------------------------------------------------------------------#
        # Backward and optimize
        net_optimizer.zero_grad()
        agent_optimizer.zero_grad()

        loss.backward()  
        net_optimizer.step()
        agent_optimizer.step()
            
    return tasks_top1.avg , tasks_losses.avg

def train_no_agent(dataset, poch, train_loader, net, net_optimizer, w0_dict):
    #Train the model
    net.train()

    total_step = len(train_loader)
    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter()

    for i, task_batch in enumerate(train_loader):
        images = task_batch[0] 
        labels = task_batch[1]    
     
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        
        images, labels = Variable(images), Variable(labels)	   
        outputs = net.forward(images, use_multitune, policy=None)
        _, predicted = torch.max(outputs.data, 1)

        correct = predicted.eq(labels.data).cpu().sum()
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
        
        existing_l2_reg = 0.0
        new_l2_reg = 0.0
                        
        for name, w in net.named_parameters():
            if 'weight' not in name:  # I don't know if that is true: I was told that Facebook regularized biases too.
                continue
            if 'downsample.1' in name:  # another bias
                continue
            if 'bn' in name:  # bn parameters
                continue
        
            if 'linear' in name:
                new_l2_reg += torch.pow(w, 2).sum()/2
#                print ('L2:', name, w.size())
            else:
                w0 = w0_dic[name].data
                w0 = w0.cuda()
                existing_l2_reg += torch.pow(w-w0, 2).sum()/2
        
        l2_reg = existing_l2_reg * 0.01 + new_l2_reg * 0.01   

        # Loss
        loss = criterion(outputs, labels)
        loss += l2_reg
        
        tasks_losses.update(loss.item(), labels.size(0))

        if i % 50 == 0:
            print ("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
                .format(epoch+1, args.nb_epochs, i+1, total_step, tasks_losses.val, tasks_top1.val, tasks_top1.avg))
       
        #---------------------------------------------------------------------#
        # Backward and optimize
        net_optimizer.zero_grad()
        agent_optimizer.zero_grad()

#        loss.requires_grad = True
        loss.backward()  
        net_optimizer.step()
        agent_optimizer.step()
            
    return tasks_top1.avg , tasks_losses.avg

def test(epoch, val_loader, net, agent, dataset, use_multitune):
    net.eval()
    agent.eval()

    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter() 

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

       	    # The policy network will be used only the original SpotTune is used.
            if not use_multitune:
                probs = agent(images)
                action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
                policy = action[:,:,1]
            
            # If using the MultiTune method, the policy network will not be used.
            if use_multitune:
                outputs = net.forward(images, use_multitune, policy=None)
            else:
                outputs = net.forward(images, use_multitune, policy)
            
            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(labels.data).cpu().sum()
            tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
        
            # Loss
            loss = criterion(outputs, labels)
            tasks_losses.update(loss.item(), labels.size(0))           

    print ("test accuracy------------------------------------------------")
    print ("Epoch [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
        .format(epoch+1, args.nb_epochs, tasks_losses.avg, tasks_top1.val, tasks_top1.avg))

    return tasks_top1.avg, tasks_losses.avg

def load_weights_to_flatresnet(source, net, num_class, dataset):
    from functools import partial
    import pickle
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
#    model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
    checkpoint = torch.load(source, map_location=lambda storage, loc: storage, pickle_module=pickle)
    net_old = checkpoint['net']
    
    store_data = []
    t = 0
    for name, m in net_old.named_modules():
        if isinstance(m, nn.Conv2d):
            store_data.append(m.weight.data)
            t += 1

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' not in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            element += 1

    element = 1
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            element += 1

    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'parallel_block' not in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
            m.running_var = store_data_rv[element].clone()
            m.running_mean = store_data_rm[element].clone()
            element += 1

    element = 1
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'parallel_block' in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
            m.running_var = store_data_rv[element].clone()
            m.running_mean = store_data_rm[element].clone()
            element += 1
    
    del net_old
    return net

def get_model(model, num_class, dataset = None):
    if model == 'resnet26':
        rnet = resnet26(num_class)
        if dataset is not None:
            if dataset == 'imagenet12':
            	source = './resnet26_pretrained.t7'
            else:
                source = './cv/' + dataset + '/' + dataset + '.t7'
        rnet = load_weights_to_flatresnet(source, rnet, num_class, dataset)
    return rnet

def load_data(directory, use_air):
    with open('./decathlon-1.0-data/' + 'decathlon_mean_std.pickle', 'rb') as handle:
        dict_mean_std = pickle.load(handle, encoding='bytes')
    
    num_classes = []
    train_loader = []
    val_loader = []
    if use_air:
        transform = transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(72),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = dict_mean_std[('aircraftmean').encode('utf-8')],
                              std = dict_mean_std[('aircraftstd').encode('utf-8')])
        ])
    else:
        transform = transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(72),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = dict_mean_std[('cifar100mean').encode('utf-8')],
                              std = dict_mean_std[('cifar100std').encode('utf-8')])
        ])
        
    traindir = os.path.join(directory, 'train')
    valdir = os.path.join(directory, 'val')
    train = torchvision.datasets.ImageFolder(traindir, transform)
    val = torchvision.datasets.ImageFolder(valdir, transform)
    
    train_loader.append(torch.utils.data.DataLoader(train, batch_size=120, 
                                           shuffle=True, num_workers=0))
    val_loader.append(torch.utils.data.DataLoader(val, batch_size=120, 
                                           shuffle=True, num_workers=0))
    num_classes.append(len(train_loader[0].dataset.classes))
    
    return train_loader, val_loader, num_classes

#####################################
# Prepare data loaders
if use_air:
    if use_multitune:
        train_loaders,val_loaders, num_classes = load_data('./decathlon-1.0-data/data/aircraft', use_air)
    else:
        train_loaders, val_loaders, num_classes = imdbfolder.prepare_data_loaders(datasets.keys(), args.datadir, args.imdbdir, True)
else:
    if use_multitune:
        train_loaders,val_loaders, num_classes = load_data('./decathlon-1.0-data/data/cifar100', use_air)
    else:
        train_loaders, val_loaders, num_classes = imdbfolder.prepare_data_loaders(datasets.keys(), args.datadir, args.imdbdir, True)

# Making a small dataset:
# These lines of code are used for making small datasets, change number_per_class to specify the number of images you want per class,
# Default is 10.
if run_small:
    number_per_class = 10    
    makeFolder('./small_dataset')
    if use_air:
        moveAllFilesinDir('./decathlon-1.0-data/data/aircraft', './small_dataset/', number_per_class)
    else:
        moveAllFilesinDir('./decathlon-1.0-data/data/cifar100', './small_dataset/', number_per_class)
    train_loaders,val_loaders, num_classes = load_data('./small_dataset', use_air)

criterion = nn.CrossEntropyLoss()

for i, dataset in enumerate(datasets.keys()):
    print (dataset) 
    pretrained_model_dir = args.ckpdir + dataset

    if not os.path.isdir(pretrained_model_dir):
        os.mkdir(pretrained_model_dir)

    results = np.zeros((4, args.nb_epochs, len(num_classes)))
    f = pretrained_model_dir + "/params.json"
    with open(f, 'w') as fh:
#        print(vars(args))
#        print(fh)
        json.dump(vars(args), fh)     

    num_class = num_classes[datasets[dataset]]
    net = get_model("resnet26", num_class, dataset = "imagenet12")
	
    # Re-initialize last one block: ********************************************************************************
    if use_multitune:
        for l in range(3,4):
            for m in net.blocks[2][l].modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
            for m in net.parallel_blocks[2][l].modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
                    
    # Extract the intitial weights transferred from ImageNet 
    w0_dic = {}
    
    for name,w in net.named_parameters():
        if 'weight' not in name:  # I don't know if that is true: I was told that Facebook regularized biases too.
            continue
        if 'downsample.1' in name:  # another bias
            continue
        if 'bn' in name:  # bn parameters
            continue
        else:
            w0 = w
            w0_dic[name] = w0

    
    agent = agent_net.resnet(sum(net.layer_config) * 2)
	
    # freeze the original blocks ************************************************************************************
    # Used when only original SpotTune code is used
    if not use_multitune:
        flag = True
        for name, m in net.named_modules():
            if isinstance(m, nn.Conv2d) and 'parallel_blocks' not in name:
                if flag is True:
                    flag = False
                else:
                    m.weight.requires_grad = False

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
        agent.cuda()
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)

    
    # Different learning rate for different layers:
    # Used only is MultiTune is used.
    if use_multitune:
        high_lr = 0.1
        low_lr = 0.01
        
        params_dict = dict(net.named_parameters())
        params = []
        #print(params_dict.keys())
        j = 0
        for key, value in reversed(list(params_dict.items())):
            if 'parallel_blocks' in key:
                if(not(key.split('.')[1].isdigit())):
                    params += [{'params':[value],'lr':args.lr, 'name': key}]  
                elif(int(key.split('.')[1]))<2:
                    params += [{'params':[value],'lr':high_lr, 'name': key}] 
                else:
                    params += [{'params':[value],'lr':low_lr, 'name': key}]   
            else:
                if(not(key.split('.')[1].isdigit())):
                    params += [{'params':[value],'lr':args.lr, 'name': key}]  
                elif(int(key.split('.')[1]))<2:
                    params += [{'params':[value],'lr':args.lr, 'name': key}] 
                else:
                    params += [{'params':[value],'lr':args.lr, 'name': key}]   
                
    
        optimizer = optim.SGD(params, momentum=0.9, weight_decay= weight_decays[dataset])
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr= args.lr, momentum=0.9, weight_decay= weight_decays[dataset])
    
    agent_optimizer = optim.SGD(agent.parameters(), lr= args.lr_agent, momentum= 0.9, weight_decay= 0.001)

    start_epoch = 0
    best_acc = 0.0
    
    epoch_accuracy = []
    total_time = 0.0
    
    '''
    lr_decay = [25, 20, 20]
#    lr_decay = [35, 30, 30]
    decay_value = [0.2, 0.4, 0.4]
#    decay_value = [0.1, 0.2, 0.2]
    lr_decay1 = 40

    for epoch in range(start_epoch, start_epoch+args.nb_epochs):
        if(epoch>1):
            adjust_learning_rate(optimizer, epoch, lr_decay, lr_decay1, decay_value)
            adjust_learning_rate(agent_optimizer, epoch, lr_decay, lr_decay1, decay_value)
            
    '''
    
#    lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

    for epoch in range(start_epoch, start_epoch+args.nb_epochs):
        adjust_learning_rate_net(optimizer, epoch, args)
        adjust_learning_rate_agent(agent_optimizer, epoch, args)

        st_time = time.time()
        if use_multitune:
            train_acc, train_loss = train_no_agent(dataset, epoch, train_loaders[datasets[dataset]], net, optimizer, w0_dic)
        else:
            train_acc, train_loss = train(dataset, epoch, train_loaders[datasets[dataset]], net, agent, optimizer, agent_optimizer, w0_dic)

        test_acc, test_loss = test(epoch, val_loaders[datasets[dataset]], net, agent, dataset, use_multitune)
        
        epoch_accuracy.append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        # Record statistics
        results[0:2,epoch,i] = [train_loss, train_acc]
        results[2:4,epoch,i] = [test_loss,test_acc]
        
        total_time += time.time()-st_time
        print('Epoch lasted {0}'.format(time.time()-st_time))
        print('Best test accuracy:', best_acc)
    
    plt.figure(figsize=(15,10))     
    plt.plot(epoch_accuracy)
    plt.ylabel('Validation Accuracy (%)')
    plt.xlabel('Number of Epoch')    
    plt.show()
    plt.savefig('epoch_accuracy.png')
    print('Total time used:', total_time/60.0)
    
    state = {
        'net': net,
        'agent': agent,
    }

    torch.save(state, pretrained_model_dir +'/' + dataset + '.t7')
    np.save(pretrained_model_dir + '/statistics', results)
