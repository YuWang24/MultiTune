import os
import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as torchdata

import agent_net 
from spottune_models import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0      
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate_net(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    if epoch >= args.step3:
        lr = args.lr * 0.001
    elif epoch >= args.step2:
        lr = args.lr * 0.01        
    elif epoch >= args.step1:
        lr = args.lr * 0.1
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def adjust_learning_rate_agent(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    if epoch >= args.step3:
        lr = args.lr_agent * 0.001
    elif epoch >= args.step2:
        lr = args.lr_agent * 0.01        
    elif epoch >= args.step1:
        lr = args.lr_agent * 0.1
    else:
        lr = args.lr_agent

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

'''
def adjust_learning_rate(optimizer, epoch, lr_decay, lr_decay1, decay_value):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
                
    for i, param_group in enumerate(optimizer.param_groups):
#        print(param_group['params'])
        lr = param_group['lr']
#        name = param_group['name']
        #print(i)
        #print(lr)
        #print(lr_decay[0])
        #print(lr_decay[1])
        #print(lr_decay[2])
#        if((not(name.split('.')[1].isdigit())) or (int(name.split('.')[1])>=(21-high_lr_layers))):
          #print(name)
          #print('hi')
        if(epoch>(lr_decay[0] + lr_decay[1] + 1)):
            #print('hi1')
            #print(int(round((epoch-(lr_decay[0] + lr_decay[1]))%lr_decay[2])))
            if(int(round(((epoch-(lr_decay[0] + lr_decay[1]))%lr_decay[2]))) == 1):
                #print('epoch: ' + str(epoch))
                #print(lr)
                #print(decay_value[2])
                lr = param_group['lr']*decay_value[2]
#                print(name)
                print(lr)
                
        elif(epoch>(lr_decay[0]+1)):
                #print('hi2')
                #print(int(round((epoch-(lr_decay[0]))%lr_decay[1])))
                if(int(round((epoch-(lr_decay[0]))%lr_decay[1])) == 1):            
                    #print('epoch: ' + str(epoch))
                    lr = param_group['lr']*decay_value[1]
#                    print(name)
                    print(lr)
                    
        else: 
            #print('hi3')
            #print(int(round(epoch%lr_decay[0])))
            if(int(round(epoch%lr_decay[0])) == 1):
                lr = param_group['lr']*decay_value[0]
#                print(name)
                print(lr)
                
##        else:
#        if(epoch>(lr_decay[0] + lr_decay[1] + 1)):
#            if(int(round(((epoch-(lr_decay[0] + lr_decay[1]))%lr_decay[2])) == 1)):
#                #print('epoch: ' + str(epoch))
#                lr = param_group['lr']*decay_value[2]
##                print(name)
#                print(lr) 
#        elif(epoch>(lr_decay[0]+1)):
#                if((int(round((epoch-(lr_decay[0]))%lr_decay[1])) == 1)):            
#                    #print('epoch: ' + str(epoch))
#                    lr = param_group['lr']*decay_value[1]
##                    print(name)
#                    print(lr)
           
        param_group['lr'] = lr
#'''       
        
def load_weights_to_flatresnet(source, net, agent,  dataset):
    checkpoint = torch.load(source)
    net_old = checkpoint['net']

    store_data = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.Conv2d):
                store_data.append(m.weight.data)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
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
        if isinstance(m, nn.BatchNorm2d):
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

    agent_old = checkpoint['agent']
    store_data = []
    for name, m in agent_old.named_modules():
        if isinstance(m, nn.Conv2d):
                store_data.append(m.weight.data)

    element = 0
    for name, m in agent.named_modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            element += 1

    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in agent_old.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in agent.named_modules():
        if isinstance(m, nn.BatchNorm2d): 
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

    agent.linear.weight.data = torch.nn.Parameter(agent_old.module.linear.weight.data.clone())
    agent.linear.bias.data = torch.nn.Parameter(agent_old.module.linear.bias.data.clone())

    net.linear.weight.data = torch.nn.Parameter(net_old.module.linear.weight.data.clone())
    net.linear.bias.data = torch.nn.Parameter(net_old.module.linear.bias.data.clone())

    del net_old
    del agent_old
    return net, agent

def get_net_and_agent(model, num_class, dataset = None):
    if model == 'resnet26':
        if dataset is not None:
            source = '../cv/' + dataset + '/' + dataset + '.t7'
            rnet = resnet26(num_class)
            agent = agent_net.resnet(sum(rnet.layer_config))

            rnet, agent = load_weights_to_flatresnet(source, rnet, agent, dataset)

            return rnet, agent