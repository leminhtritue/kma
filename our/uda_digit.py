import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from data_load import mnist, svhn, usps
import collections

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def digit_load(args): 
    train_bs = args.batch_size
    if args.dset == 's2m':
        train_source = svhn.SVHN('./data/svhn/', split='train', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        test_source = svhn.SVHN('./data/svhn/', split='test', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))  
        train_target = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))      
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
    elif args.dset == 'u2m':
        train_source = usps.USPS('./data/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_source = usps.USPS('./data/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))    
        train_target = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))    
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    elif args.dset == 'm2u':
        train_source = mnist.MNIST('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_source = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))

        train_target = usps.USPS_idx('./data/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_target = usps.USPS('./data/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source, batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["source_te"] = DataLoader(test_source, batch_size=train_bs*2, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["target_te"] = DataLoader(train_target, batch_size=train_bs, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs*2, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    return dset_loaders

def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent

def get_hyperplane(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    return all_output

def cal_acc_plot(loader, netF, ouput_name, label_name):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netF(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    print(all_output.shape)
    print(all_label.shape)
    all_output_np = all_output.numpy()
    all_label_np = all_label.numpy()
    np.save(ouput_name, all_output_np)
    np.save(label_name, all_label_np)
    # _, predict = torch.max(all_output, 1)
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    # return accuracy*100, mean_ent

def get_feature_label(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netB(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    return all_output, all_label

def train_source(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, gamma = args.gamma, bottleneck_dim=args.bottleneck).cuda()
    # netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type="linear", class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    # netF.train()
    # netB.train()
    # netC.train()
    total_loss = 0.0
    count_loss = 0

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source))) #64x10
        classifier_loss = loss.KernelSource(num_classes=args.class_num, alpha=args.smooth)(outputs_source, labels_source, netC) 
        # classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)  
        total_loss += classifier_loss
        count_loss += 1           
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_tr, _ = cal_acc(dset_loaders['source_tr'], netF, netB, netC)
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy source (train/test) = {:.2f}%/ {:.2f}%, Loss = {:.2f}'.format(args.dset, iter_num, max_iter, acc_s_tr, acc_s_te, total_loss/count_loss)
            total_loss = 0.0
            count_loss = 0
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
            # netF.train()
            # netB.train()
            # netC.train()                

    source_train_data, source_train_label = get_feature_label(dset_loaders['source_tr'], netF, netB, netC)
    source_test_data, source_test_label = get_feature_label(dset_loaders['source_te'], netF, netB, netC)
    print(source_train_data.shape, source_train_label.shape, source_test_data.shape, source_test_label.shape)
    all_source_data = torch.cat((source_train_data, source_test_data), 0)
    all_source_label = torch.cat((source_train_label, source_test_label), 0)
    print(all_source_data.shape, all_source_label.shape)

    a_count = 0
    for i in range(args.class_num):
    	start_test = True
    	cur_data = all_source_data[all_source_label == i]
    	cur_shape0 = cur_data.shape[0]
    	a_count += cur_shape0
    	cur_mean = cur_data.mean(dim=0)
    	print(cur_mean[:5])
    	if start_test:
    		mean_out = cur_mean
    		start_test = False
    	else:
    		mean_out = torch.cat((mean_out, cur_mean), 0)
    print(mean_out.shape)
    print(mean_out[:,:5])
    print(a_count)
    sys.exit()

    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))

    return netF, netB, netC

def test_target(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, gamma = args.gamma, bottleneck_dim=args.bottleneck).cuda()
    # netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type="linear", class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
    log_str = 'Task: {}, Source model accuracy on target test = {:.2f}%'.format(args.dset, acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

def test_dataset(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, gamma = args.gamma, bottleneck_dim=args.bottleneck).cuda()
    # netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type="linear", class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    # args.modelpath = args.output_dir + '/source_F.pt'   
    # netF.load_state_dict(torch.load(args.modelpath))
    # args.modelpath = args.output_dir + '/source_B.pt'   
    # netB.load_state_dict(torch.load(args.modelpath))
    # args.modelpath = args.output_dir + '/source_C.pt'   
    # netC.load_state_dict(torch.load(args.modelpath))

    args.modelpath = args.output_dir + '/target_F_par_0.1.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/target_B_par_0.1.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/target_C_par_0.1.pt'   
    netC.load_state_dict(torch.load(args.modelpath))

    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders[args.dataset], netF, netB, netC)
    log_str = 'Task: {}, Source model accuracy on {} = {:.2f}%'.format(args.dset, args.dataset, acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

def extract_hyperplane(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, gamma = args.gamma, bottleneck_dim=args.bottleneck).cuda()
    # netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type="linear", class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    # args.modelpath = args.output_dir + '/source_F.pt'   
    # netF.load_state_dict(torch.load(args.modelpath))
    # args.modelpath = args.output_dir + '/source_B.pt'   
    # netB.load_state_dict(torch.load(args.modelpath))
    # args.modelpath = args.output_dir + '/source_C.pt'   
    # netC.load_state_dict(torch.load(args.modelpath))

    args.modelpath = args.output_dir + '/target_F_par_0.1.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/target_B_par_0.1.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/target_C_par_0.1.pt'   
    netC.load_state_dict(torch.load(args.modelpath))

    netF.eval()
    netB.eval()
    netC.eval()

    hyperplane_score = get_hyperplane(dset_loaders[args.dataset], netF, netB, netC)
    hyperplane_score_abs = torch.abs(hyperplane_score)
    t = netC.get_weight()
    hyperplane_score_abs = hyperplane_score_abs.cuda()
    t = t * t
    t = t.sum(dim = 1)
    t = hyperplane_score_abs / t
    _, predict = torch.min(t, 1)
    print(collections.Counter(predict.cpu().numpy()))

    hyperplane_score[hyperplane_score < 0] = 0
    hyperplane_score[hyperplane_score > 0] = 1
    hyperplane_score = hyperplane_score.sum(dim = 1)
    print(collections.Counter(hyperplane_score.numpy()))

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def train_target(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, gamma = args.gamma, bottleneck_dim=args.bottleneck).cuda()
    # netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type="linear", class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'    
    netC.load_state_dict(torch.load(args.modelpath))
    
    # for k, v in netC.named_parameters():
    #     v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    # for k, v in netC.named_parameters():
    #     param_group += [{'params': v, 'lr': args.lr}] 

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = len(dset_loaders["target"])
    # interval_iter = max_iter // args.interval
    iter_num = 0

    # netF.train()
    # netB.train()
    # netC.train()

    classifier_loss_total = 0.0
    classifier_loss_count = 0
    entropy_loss_total = 0.0
    entropy_loss_count = 0
    costlog_loss_total = 0.0
    costlog_loss_count = 0
    costs_loss_total = 0.0
    costs_loss_count = 0
    right_sample_count = 0
    sum_sample = 0
    start_output = True

    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        # if iter_num % interval_iter == 0 and args.cls_par > 0:
        #     netF.eval()
        #     netB.eval()
        #     netC.eval()
        #     mem_label = obtain_label(dset_loaders['target_te'], netF, netB, netC, args)
        #     mem_label = torch.from_numpy(mem_label).cuda()
        #     netF.train()
        #     netB.train()
        #     netC.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_test = inputs_test.cuda()
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        # if args.cls_par > 0:
        #     pred = mem_label[tar_idx]
        #     classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_test, pred)
        # else:
        #     classifier_loss = torch.tensor(0.0).cuda()


        # softmax_out = nn.Softmax(dim=1)(outputs_test)
        # entropy_loss = torch.mean(loss.Entropy(softmax_out))
        # classifier_loss = entropy_loss

        mark_max = torch.zeros(outputs_test.size()).cuda()
        mark_zeros = torch.zeros(outputs_test.size()).cuda()

        # outputs_test_max = torch.maximum(outputs_test, mark_zeros)
        outputs_test_max = outputs_test
        
        for i in range(args.class_num):
            mark_max[:,i] = torch.max(torch.cat((outputs_test_max[:, :i],outputs_test_max[:, i+1:]), dim = 1), dim = 1).values        

        # cost_s = nn.Softmax(dim=1)(outputs_test_max - mark_max)
        cost_s = -torch.maximum(outputs_test_max - mark_max, mark_zeros)
        

        softmax_out = nn.Softmax(dim=1)(outputs_test)
        cost_log = -torch.log(softmax_out + 1e-5)
        cost = cost_log + cost_s
        # print(cost_s.mean())
        # print(cost_log.mean())

        entropy_raw = softmax_out * cost
        entropy_raw = torch.sum(entropy_raw, dim=1)         
        # entropy_raw = loss.Entropy(softmax_out)
        entropy_loss = torch.mean(entropy_raw)
        if args.gent > 0:
            msoftmax = softmax_out.mean(dim=0)
            entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

        im_loss = entropy_loss * args.ent_par
        classifier_loss = im_loss

        # entropy_loss = loss.Entropy(softmax_score).mean()      
        # div_loss = -loss.Entropy_1D(softmax_score.mean(dim = 0))
        # classifier_loss = entropy_loss + div_loss
        # classifier_loss = entropy_loss

        classifier_loss_total += classifier_loss
        classifier_loss_count += 1   
        entropy_loss_total += entropy_loss
        entropy_loss_count += 1 
        costlog_loss_total += cost_log.mean()
        costlog_loss_count += 1  
        costs_loss_total += cost_s.mean()
        costs_loss_count += 1  

        max_hyperplane = outputs_test_max.max(dim=1).values       
        max_hyperplane[max_hyperplane > 0] = 1
        max_hyperplane[max_hyperplane < 0] = 0
        right_sample_count += max_hyperplane.sum()
        sum_sample += outputs_test_max.shape[0]

        if (start_output):
            all_output = outputs_test.float().cpu()
            start_output = False
        else:
            all_output = torch.cat((all_output, outputs_test.float().cpu()), 0)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            _, predict = torch.max(all_output, 1)
            acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
            acc_tr, _ = cal_acc(dset_loaders['target_te'], netF, netB, netC)
            log_str = 'Iter:{}/{}; Loss (entropy): {:.2f}, Cost (s/logp) = {:.2f} / {:.2f}, Accuracy target (train/test) = {:.2f}% / {:.2f}%, moved samples: {}/{}.'.format(iter_num, max_iter, \
            	entropy_loss_total/entropy_loss_count, costs_loss_total/costs_loss_count, costlog_loss_total/costlog_loss_count, acc_tr, acc, right_sample_count, sum_sample)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
            print(collections.Counter(predict.numpy()))
            print()

            classifier_loss_total = 0.0
            classifier_loss_count = 0
            entropy_loss_total = 0.0
            entropy_loss_count = 0
            div_loss_total = 0.0
            div_loss_count = 0
            right_sample_count = 0
            sum_sample = 0
            start_output = True

            # netF.train()
            # netB.train()
            # netC.train()

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    # cal_acc_plot(dset_loaders['source_te'], netF, "train_data", "train_label")
    # cal_acc_plot(dset_loaders['test'], netF, "test_data", "test_label")
    return netF, netB, netC

def obtain_label(loader, netF, netB, netC, args, c=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    return pred_label.astype('int')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=2, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='s2m', choices=['u2m', 'm2u','s2m'])
    parser.add_argument('--dataset', type=str, default='source_te')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--gent', type=float, default=0.1)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.5)
    args = parser.parse_args()
    args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.output_dir = osp.join(args.output, 'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):
        args.out_file = open(osp.join(args.output_dir, 'log_src.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_source(args)
        test_target(args)

    args.savename = 'par_' + str(args.cls_par)
    args.out_file = open(osp.join(args.output_dir, 'log_tar_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    test_target(args)
    train_target(args)

    # test_dataset(args)
    # extract_hyperplane(args)