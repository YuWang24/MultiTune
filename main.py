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
import models
import agent_net

from utils import *
from gumbel_softmax import *
import pickle
#import dataSplitGeneral_cars_allsizes

#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

parser = argparse.ArgumentParser(description='PyTorch SpotTune')

parser.add_argument('--nb_epochs', default=110, type=int, help='nb epochs')
#parser.add_argument('--nb_epochs', default=50, type=int, help='nb epochs')

parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate of net')
parser.add_argument('--lr_agent', default=0.01, type=float, help='initial learning rate of agent')

parser.add_argument('--datadir', default='./decathlon-1.0-data/', help='folder containing data folder')
parser.add_argument('--imdbdir', default='./decathlon-1.0-devkit/decathlon-1.0/annotations/', help='annotation folder')
parser.add_argument('--ckpdir', default='./cv/', help='folder saving checkpoint')

parser.add_argument('--seed', default=0, type=int, help='seed')

parser.add_argument('--step1', default=40, type=int, help='nb epochs before first lr decrease')
parser.add_argument('--step2', default=60, type=int, help='nb epochs before second lr decrease')
parser.add_argument('--step3', default=80, type=int, help='nb epochs before third lr decrease')

#parser.add_argument('--step1', default=10, type=int, help='nb epochs before first lr decrease')
#parser.add_argument('--step2', default=20, type=int, help='nb epochs before second lr decrease')
#parser.add_argument('--step3', default=30, type=int, help='nb epochs before third lr decrease')

args = parser.parse_args()

weight_decays = [
     ("car",0.0005)
#     ("imagenet12", 0.0005)   
#    ("aircraft", 0.0005),
#    ("cifar100", 0.0),
#    ("daimlerpedcls", 0.0005),
#    ("dtd", 0.0),
#    ("gtsrb", 0.0),
#    ("omniglot", 0.0005),
#    ("svhn", 0.0),
#    ("ucf101", 0.0005),
#    ("vgg-flowers", 0.0001),
#    ("imagenet12", 0.0001)
    ]

datasets = [
     ("car",0)
#     ("imagenet12", 0)  
#    ("aircraft", 0),
#    ("cifar100", 0),
#    ("daimlerpedcls", 2),
#    ("dtd", 3),
#    ("gtsrb", 4),
#    ("omniglot", 5),
#    ("svhn", 6),
#    ("ucf101", 7),
#    ("vgg-flowers", 8)
    ]

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
    
#    policy_matrix_top = torch.zeros([26, 128, 12], dtype=torch.float64, device=0)
#    policy_matrix_bottom = torch.zeros([6, 12], dtype=torch.float64, device=0)
    
    
    for i, task_batch in enumerate(train_loader):
        images = task_batch[0] 
        labels = task_batch[1]    
        
#        print(labels.shape)
#        print(images)
#        print(images.shape)
#        print(labels)
        
        if use_cuda:
#            images = torch.from_numpy(images)
#            labels = torch.from_numpy(labels)
#            images = images.float()
#            labels = labels.float()
            images, labels = images.cuda(), labels.cuda()
        
        images, labels = Variable(images), Variable(labels)	   
        
#        with torch.no_grad():
        probs = agent(images)

        action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
        policy = action[:,:,1]
        
#        if i!=26:
#            policy_matrix_top[i] = policy
#        else:
#            policy_matrix_bottom = policy
        
#        print(policy.shape)
        
#        with torch.no_grad():
        outputs = net.forward(images, policy)
        _, predicted = torch.max(outputs.data, 1)
#        print(labels.data.int())
#        print(predicted)
        correct = predicted.eq(labels.data).cpu().sum()
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
#        '''
#        weight_basic = torch.zeros([len(labels), 24], dtype=torch.float64, device=0)
#        print(len(labels))
#        weight_parallel = torch.zeros([len(labels), 24], dtype=torch.float64, device=0)
        
        
        existing_l2_reg = 0.0
        new_l2_reg = 0.0
                        
        '''
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
                # if I didn't misunderstand,
                # pretrained_weights[name] is a `Parameter`, which can be trained;
                # pretrained_weights[name].data is a `Tensor`, which is considered as a constant.
                # So here we just want to use w0 as constant and train w.
                # print type(w0), type(w),
#                print ('L2-SP:', name, w.size())
                existing_l2_reg += torch.pow(w-w0, 2).sum()/2
        
        l2_reg = existing_l2_reg * 0.01 + new_l2_reg * 0.01
        '''      
        '''
        for b in range(len(labels)):
            count_p = 0
            count_b = 0
            for name, w in net.named_parameters():
                if name == 'conv1.weight':
                    w0 = w0_dic[name]
                    w0 = w0.cuda()
                    existing_l2_reg += torch.pow(w-w0, 2).sum()/2
                elif 'linear' in name:
                        new_l2_reg += torch.pow(w, 2).sum()/2
                elif 'parallel_blocks' in name:
                    if 'weight' not in name:  # I don't know if that is true: I was told that Facebook regularized biases too.
                        continue
                    if 'downsample.1' in name:  # another bias
                        continue
                    if 'bn' in name:  # bn parameters
                        continue
                    else:
                        w0 = w0_dic[name]
                        w0 = w0.cuda()
                        # if I didn't misunderstand,
                        # pretrained_weights[name] is a `Parameter`, which can be trained;
                        # pretrained_weights[name].data is a `Tensor`, which is considered as a constant.
                        # So here we just want to use w0 as constant and train w.
                        # print type(w0), type(w),
        #                print ('L2-SP:', name, w.size())
                        weight_parallel[b][count_p] = torch.pow(w-w0, 2).sum()/2
                        count_p+=1
                else:
                    if 'weight' not in name:  # I don't know if that is true: I was told that Facebook regularized biases too.
                        continue
                    if 'downsample.1' in name:  # another bias
                        continue
                    if 'bn' in name:  # bn parameters
                        continue
                    else:
                        w0 = w0_dic[name]
                        w0 = w0.cuda()
                        # if I didn't misunderstand,
                        # pretrained_weights[name] is a `Parameter`, which can be trained;
                        # pretrained_weights[name].data is a `Tensor`, which is considered as a constant.
                        # So here we just want to use w0 as constant and train w.
                        # print type(w0), type(w),
        #                print ('L2-SP:', name, w.size())
#                        print(name)
                        weight_basic[b][count_b] = torch.pow(w-w0, 2).sum()/2
                        count_b+=1
        
#        print('1', weight_basic)
#        print('2', weight_parallel.shape)
        
        
        for po in range(len(policy)):
            count = 0
            for p in policy[po]:
                if p.item() == 0:
                    existing_l2_reg += weight_basic[po][count].float()
                    existing_l2_reg += weight_basic[po][count+1].float()
                    count+=2
                else:
#                    print(weight_parallel[po][count])
#                    print('po:', po)
#                    print('count:', count)
                    existing_l2_reg += weight_parallel[po][count].float()
                    existing_l2_reg += weight_parallel[po][count+1].float()
                    count+=2
                    
        l2_reg = existing_l2_reg * 0.01 + new_l2_reg * 0.01
        '''
        
        
        # Loss
        loss = criterion(outputs, labels)
#        loss += l2_reg
        
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

def train_no_agent(dataset, poch, train_loader, net, net_optimizer):
    #Train the model
    net.train()
#    agent.train()

    total_step = len(train_loader)
    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter()

    for i, task_batch in enumerate(train_loader):
        images = task_batch[0] 
        labels = task_batch[1]    
        
#        print(images)
#        print(images.shape)
#        print(labels)
        
        if use_cuda:
#            images = torch.from_numpy(images)
#            labels = torch.from_numpy(labels)
#            images = images.float()
#            labels = labels.float()
            images, labels = images.cuda(), labels.cuda()
        
        images, labels = Variable(images), Variable(labels)	   
        
#        with torch.no_grad():
#        probs = agent(images)

#        action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
#        policy = action[:,:,1]
        
#        with torch.no_grad():
        outputs = net.forward(images, policy=None)
        _, predicted = torch.max(outputs.data, 1)
#        print(labels.data.int())
#        print(predicted)
        correct = predicted.eq(labels.data).cpu().sum()
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))

        # Loss
        loss = criterion(outputs, labels)
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

def test(epoch, val_loader, net, agent, dataset):
    net.eval()
    agent.eval()

    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter() 

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

       	    probs = agent(images)
            action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
            policy = action[:,:,1]
            outputs = net.forward(images, policy)
            
            #Test without agent net
#            outputs = net.forward(images, policy=None)

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
    
#    checkpoint = torch.load(source, encoding='iso-8859-1')
#    source = source.encode('utf-8')
#    
    from functools import partial
    import pickle
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
#    model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
    checkpoint = torch.load(source, map_location=lambda storage, loc: storage, pickle_module=pickle)
    
#    print(checkpoint.keys())
    
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

def load_data(directory):
    
    with open('./decathlon-1.0-data/' + 'decathlon_mean_std.pickle', 'rb') as handle:
        dict_mean_std = pickle.load(handle, encoding='bytes')
#        print(dict_mean_std)
    
    num_classes = []
    train_loader = []
    val_loader = []
    
    transform = transforms.Compose([
#    transforms.RandomSizedCrop(224),
    transforms.Resize(72),
    transforms.CenterCrop(72),
#    transforms.RandomResizedCrop(72),
#    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = dict_mean_std[('aircraftmean').encode('utf-8')],
                         std = dict_mean_std[('aircraftstd').encode('utf-8')])
    ])
    traindir = os.path.join(directory, 'train')
    valdir = os.path.join(directory, 'val')
    train = torchvision.datasets.ImageFolder(traindir, transform)
    val = torchvision.datasets.ImageFolder(valdir, transform)
    
    train_loader.append(torch.utils.data.DataLoader(train, batch_size=4, 
                                           shuffle=True, num_workers=0))
    val_loader.append(torch.utils.data.DataLoader(val, batch_size=4, 
                                           shuffle=True, num_workers=0))
    num_classes.append(len(train_loader[0].dataset.classes))
    
    return train_loader, val_loader, num_classes

def load_car(directory):
    num_classes = []
    train_loader = []
    test_loader = []
    
    traindir = os.path.join(directory, 'train')
    testdir = os.path.join(directory, 'test')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    train_transform = transforms.Compose([
#    transforms.Scale(342),
#    transforms.CenterCrop(240),
#    transforms.Resize((400, 400)),
    transforms.Resize(72),
    transforms.CenterCrop(72),
    transforms.RandomResizedCrop(72),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    normalize
    ])
    
    test_transform = transforms.Compose([
#    transforms.Scale(342),
#    transforms.CenterCrop(240),
#    transforms.Resize((400, 400)),
    transforms.Resize(72),
    transforms.CenterCrop(72),
    transforms.ToTensor(),
#    transforms.RandomHorizontalFlip(),
    normalize
    ])
    
    train = torchvision.datasets.ImageFolder(traindir, train_transform)
    test = torchvision.datasets.ImageFolder(testdir, test_transform)
    #train_sampler = SubsetRandomSampler(train_idx)
    train_loader.append(torch.utils.data.DataLoader(
            train, batch_size=120, shuffle=True,
            num_workers=0, pin_memory=True)) #, sampler=train_sampler)

        #test_sampler = SubsetRandomSampler(test_idx)
    test_loader.append(torch.utils.data.DataLoader(test,
            batch_size=120, shuffle=False,
            num_workers=0, pin_memory=True))#, sampler=test_sampler)
    num_classes.append(len(train_loader[0].dataset.classes))
    
    return train_loader, test_loader, num_classes
#####################################
# Prepare data loaders
#train_loaders, val_loaders, num_classes = imdbfolder.prepare_data_loaders(datasets.keys(), args.datadir, args.imdbdir, True)

train_loaders,val_loaders, num_classes = load_data('./air')

#train_loaders,val_loaders, num_classes = load_car('./car')

#train_loaders,val_loaders, num_classes = load_car('./stanford_car/car_data/car_data')

#train_loaders,val_loaders, num_classes = load_data('./decathlon-1.0-data/data/aircraft')
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
	
    # Re-initialize last one block:
#    for l in range(3,4):
#        for m in net.blocks[2][l].modules():
#                if isinstance(m, nn.Conv2d):
#                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                    m.weight.data.normal_(0, math.sqrt(2. / n))
#                elif isinstance(m, nn.BatchNorm2d):
#                    m.weight.data.fill_(1)
#                    m.bias.data.zero_()
#        for m in net.parallel_blocks[2][l].modules():
#                if isinstance(m, nn.Conv2d):
#                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                    m.weight.data.normal_(0, math.sqrt(2. / n))
#                elif isinstance(m, nn.BatchNorm2d):
#                    m.weight.data.fill_(1)
#                    m.bias.data.zero_()
                    
#    '''
    w0_dic = {}
    
    for name,w in net.named_parameters():
        if 'weight' not in name:  # I don't know if that is true: I was told that Facebook regularized biases too.
            continue
        if 'downsample.1' in name:  # another bias
            continue
        if 'bn' in name:  # bn parameters
            continue
    
#        if 'fc' in name:
#            new_l2_reg += torch.pow(w, 2).sum()
#            print 'L2:', name, w.size()
        else:
            w0 = w
            # if I didn't misunderstand,
            # pretrained_weights[name] is a `Parameter`, which can be trained;
            # pretrained_weights[name].data is a `Tensor`, which is considered as a constant.
            # So here we just want to use w0 as constant and train w.
            # print type(w0), type(w),
            w0_dic[name] = w0
#            print 'L2-SP:', name, w.size()
#            existing_l2_reg += torch.pow(w-w0, 2).sum()
#    print(w0_dic.keys())
#    '''
    
    agent = agent_net.resnet(sum(net.layer_config) * 2)
	
    # freeze the original blocks
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
#        cudnn.enabled = False
        
        torch.cuda.manual_seed_all(args.seed)
        #net = nn.DataParallel(net)
        #agent = nn.DataParallel(agent)
    '''
    high_lr = 0.2
    low_lr = 0.05
    
    params_dict = dict(net.named_parameters())
    params = []
    #print(params_dict.keys())
    j = 0
    for key, value in reversed(list(params_dict.items())):  
        j+=1
#        print(key.split('.')[1])
        if(not(key.split('.')[1].isdigit())):
            print(1)
            params += [{'params':[value],'lr':0.1, 'name': key}]  
        elif(int(key.split('.')[1]))<1:
            print(2)
            params += [{'params':[value],'lr':high_lr, 'name': key}] 
        else:
            print(3)
            params += [{'params':[value],'lr':low_lr, 'name': key}]   
    print(j)
    

    optimizer = optim.SGD(params, lr= args.lr, momentum=0.9, weight_decay= weight_decays[dataset])
    '''
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr= args.lr, momentum=0.9, weight_decay= weight_decays[dataset])
    agent_optimizer = optim.SGD(agent.parameters(), lr= args.lr_agent, momentum= 0.9, weight_decay= 0.001)

    start_epoch = 0
    best_acc = 0.0
    
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
    epoch_accuracy = []
    for epoch in range(start_epoch, start_epoch+args.nb_epochs):
        adjust_learning_rate_net(optimizer, epoch, args)
        adjust_learning_rate_agent(agent_optimizer, epoch, args)

        st_time = time.time()
        train_acc, train_loss = train(dataset, epoch, train_loaders[datasets[dataset]], net, agent, optimizer, agent_optimizer, w0_dic)
#        train_acc, train_loss = train_no_agent(dataset, epoch, train_loaders[datasets[dataset]], net, optimizer)
        
        test_acc, test_loss = test(epoch, val_loaders[datasets[dataset]], net, agent, dataset)
        
        epoch_accuracy.append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        # Record statistics
        results[0:2,epoch,i] = [train_loss, train_acc]
        results[2:4,epoch,i] = [test_loss,test_acc]

        print('Epoch lasted {0}'.format(time.time()-st_time))
        print('Best test accuracy:', best_acc)

    plt.plot(epoch_accuracy)    
    plt.show()
    plt.savefig('epoch_accuracy.png')
    
    state = {
        'net': net,
        'agent': agent,
    }

    torch.save(state, pretrained_model_dir +'/' + dataset + '.t7')
    np.save(pretrained_model_dir + '/statistics', results)
