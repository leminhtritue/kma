import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

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

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
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

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def cal_accrf(loader, netF, netB, netBRF, netCRF, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netCRF(netBRF(netB(netF(inputs))))
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

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def cal_acc2(loader, netF, netB, netC, netBRF, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netBRF(netB(netF(inputs))))
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

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def normalize_perturbation(d):
    d_ = d.view(d.size()[0], -1)
    eps = d.new_tensor(1e-12)
    output = d / torch.sqrt(torch.max((d_**2).sum(dim = -1), eps)[0] )
    return output

class KLDivWithLogits(nn.Module):

    def __init__(self):

        super(KLDivWithLogits, self).__init__()

        self.kl = nn.KLDivLoss(size_average=False, reduce=True)
        self.logsoftmax = nn.LogSoftmax(dim = 1)
        self.softmax = nn.Softmax(dim = 1)


    def forward(self, x, y):

        log_p = self.logsoftmax(x)
        q     = self.softmax(y)

        return self.kl(log_p, q) / x.size()[0]

def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netBRF = network.feat_bootleneck_rf(nrf=args.nrf, type=args.classifier, gamma = args.gamma, bottleneck_dim=args.bottleneck).cuda()
    netCRF = network.feat_classifier_rf(nrf=args.nrf, type=args.layer_rf, class_num = args.class_num).cuda()
    
    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))

    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))

    if args.fromShot == 0.0:
        modelpath = args.output_dir_src + '/source_BRF.pt'    
        netBRF.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir_src + '/source_CRF.pt'    
        netCRF.load_state_dict(torch.load(modelpath))

    param_group = []
    if args.train_c == 0.0:
        netC.eval()
        for k, v in netC.named_parameters():
            v.requires_grad = False
    else:
        for k, v in netC.named_parameters():
            if args.lr_decayc > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decayc}]
            else:
                v.requires_grad = False        

    if args.train_rf == 0.0:
        netCRF.eval()
        for k, v in netCRF.named_parameters():
            v.requires_grad = False
    else:
        for k, v in netCRF.named_parameters():
            if args.lr_decay2 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
            else:
                v.requires_grad = False   


    
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    for k, v in netBRF.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    # interval_iter = max_iter // args.interval
    interval_iter = len(dset_loaders["target"])
    iter_num = 0

    classifier_loss_total = 0.0
    classifier_loss_count = 0
    # entropy_loss_total = 0.0
    # entropy_loss_count = 0
    # costlog_loss_total = 0.0
    # costlog_loss_count = 0
    # costs_loss_total = 0.0
    # costs_loss_count = 0
    # costdist_loss_total = 0.0
    # costdist_loss_count = 0
    # right_sample_count = 0
    # sum_sample = 0
    # start_output = True

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            netBRF.eval()
            if args.train_c != 0.0:
                netC.eval()
            if args.train_rf != 0.0:
                netCRF.eval()
            mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()
            netBRF.train()
            if args.train_c != 0.0:
                netC.train()
            if args.train_rf != 0.0:
                netCRF.train()

        if iter_num % interval_iter == 0 and args.cls_parrf > 0:
            netF.eval()
            netB.eval()
            netBRF.eval()
            if args.train_c != 0.0:
                netC.eval()
            if args.train_rf != 0.0:
                netCRF.eval()
            mem_label_rf = obtain_labelrf(dset_loaders['test'], netF, netB, netBRF, netCRF, args)
            mem_label_rf = torch.from_numpy(mem_label_rf).cuda()
            netF.train()
            netB.train()
            netBRF.train()
            if args.train_c != 0.0:
                netC.train()
            if args.train_rf != 0.0:
                netCRF.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        outputs_test_rf = netCRF(netBRF(features_test.detach()))

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.cls_parrf > 0:
            pred_rf = mem_label_rf[tar_idx]
            classifier_loss_rf = args.cls_parrf * nn.CrossEntropyLoss()(outputs_test_rf, pred_rf)
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss_rf *= 0
            classifier_loss += classifier_loss_rf


        softmax_out = nn.Softmax(dim=1)(outputs_test)
        if args.ent:
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss



        mark_max = torch.zeros(outputs_test_rf.size()).cuda()
        
        mark_zeros = torch.zeros(outputs_test_rf.size()).cuda()
        if (args.max_zero > 0.0):
            outputs_test_max = torch.maximum(outputs_test_rf, mark_zeros)
        else:
            outputs_test_max = outputs_test_rf

        for i in range(args.class_num):
            mark_max[:,i] = torch.max(torch.cat((outputs_test_max[:, :i],outputs_test_max[:, i+1:]), dim = 1), dim = 1).values        
        cost_s = outputs_test_max - mark_max
   
        softmax_si = nn.Softmax(dim=1)(cost_s)

        if args.alpha_rf > 0:

            entropy_si = -(softmax_out * torch.log(softmax_si + 1e-5))
            entropy_si = torch.sum(entropy_si, dim=1)
            entropy_si_loss = torch.mean(entropy_si)
            classifier_loss += args.alpha_rf * entropy_si_loss

        if args.alpha_rfen > 0:
            entropy_loss_rf = torch.mean(loss.Entropy(softmax_si))

            msoftmax_rf = softmax_si.mean(dim=0)
            gentropy_loss_rf = torch.sum(-msoftmax_rf * torch.log(msoftmax_rf + args.epsilon))
            entropy_loss_rf -= gentropy_loss_rf
            im_loss_rf = entropy_loss_rf * args.alpha_rfen
            classifier_loss += im_loss_rf

        if (args.w_vat > 0):
        	eps = (torch.randn(size=inputs_test.size())).type(inputs_test.type())
        	eps = 1e-6 * normalize_perturbation(eps)
        	eps.requires_grad = True
        	outputs_source_adv_eps = netC(netB(netF(inputs_test + eps)))
        	loss_func_nll = KLDivWithLogits()
        	loss_eps  = loss_func_nll(outputs_source_adv_eps, outputs_test.detach())
        	loss_eps.backward()
        	eps_adv = eps.grad
        	eps_adv = normalize_perturbation(eps_adv)
        	inputs_source_adv = inputs_test + args.radius * eps_adv
        	output_source_adv = netC(netB(netF(inputs_source_adv.detach())))
        	loss_vat     = loss_func_nll(output_source_adv, outputs_test.detach())
        	classifier_loss += args.w_vat * loss_vat

        classifier_loss_total += classifier_loss
        classifier_loss_count += 1   
        # entropy_loss_total += entropy_loss
        # entropy_loss_count += 1 
        # costlog_loss_total += 0
        # costlog_loss_count += 1  
        # costs_loss_total += 0
        # costs_loss_count += 1  
        # costdist_loss_total += 0
        # costdist_loss_count += 1 

        # max_hyperplane = outputs_test.max(dim=1).values       
        # max_hyperplane[max_hyperplane > 0] = 1
        # max_hyperplane[max_hyperplane < 0] = 0
        # right_sample_count += max_hyperplane.sum()
        # sum_sample += outputs_test.shape[0]

        # if (start_output):
        #     all_output = outputs_test.float().cpu()
        #     start_output = False
        # else:
        #     all_output = torch.cat((all_output, outputs_test.float().cpu()), 0)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netBRF.eval()
            if args.train_c != 0.0:
                netC.eval()
            if args.train_rf != 0.0:
                netCRF.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                # _, predict = torch.max(all_output, 1)
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                acc_s_tr, _ = cal_acc(dset_loaders['target'], netF, netB, netC, False)
                acc_s_te_rf, _ = cal_accrf(dset_loaders['test'], netF, netB, netBRF, netCRF, False)
                acc_s_tr_rf, _ = cal_accrf(dset_loaders['target'], netF, netB, netBRF, netCRF, False)

                # log_str = 'Task: {}, Iter:{}/{}; Loss : {:.2f}, , Accuracy target (train/test) = {:.2f}% / {:.2f}%, moved samples: {}/{}.'.format(args.name, iter_num, max_iter, \
                # classifier_loss_total/classifier_loss_count, acc_s_tr, acc_s_te, right_sample_count, sum_sample)
                log_str = 'Task: {}, Iter:{}/{}; Loss : {:.2f}, , Accuracy target (train/test/trainrf/testrf) = {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}%.'.format(args.name, iter_num, max_iter, \
                classifier_loss_total/classifier_loss_count, acc_s_tr, acc_s_te, acc_s_tr_rf, acc_s_te_rf)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te < 50:
                return netF, netB, netC, (acc_s_te + 100)

            classifier_loss_total = 0.0
            classifier_loss_count = 0
            # entropy_loss_total = 0.0
            # entropy_loss_count = 0
            # costs_loss_total = 0.0
            # costs_loss_count = 0
            # costdist_loss_total = 0.0
            # costdist_loss_count = 0
            # costlog_loss_total = 0.0
            # costlog_loss_count = 0
            # right_sample_count = 0
            # sum_sample = 0
            # start_output = True

            netF.train()
            netB.train()
            netBRF.train()
            if args.train_c != 0.0:
                netC.train()
            if args.train_rf != 0.0:
                netCRF.train()

    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        torch.save(netBRF.state_dict(), osp.join(args.output_dir, "target_BRF_" + args.savename + ".pt"))
        torch.save(netCRF.state_dict(), osp.join(args.output_dir, "target_CRF_" + args.savename + ".pt"))
        
    return netF, netB, netC, acc_s_te

def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB, netC)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name, acc_os2, acc_os1, acc_unknown)
    else:
        if args.dset=='VISDA-C':
            acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
        else:
            acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = '\nTraining:, Task: {}, Accuracy = {:.2f}%'.format(args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def train_target2(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
        netF_c = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  
        netF_c = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netB_c = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer_2, class_num = args.class_num, bottleneck_dim=args.nrf).cuda()

    netBRF = network.feat_bootleneck_rf(nrf=args.nrf, type=args.classifier, gamma = args.gamma, bottleneck_dim=args.bottleneck).cuda()
    netBRF_c = network.feat_bootleneck_rf(nrf=args.nrf, type=args.classifier, gamma = args.gamma, bottleneck_dim=args.bottleneck).cuda()
    netCRF = network.feat_classifier_rf(nrf=args.nrf, type=args.layer_rf, class_num = args.class_num).cuda()
    netCRF_c = network.feat_classifier_rf(nrf=args.nrf, type=args.layer_rf, class_num = args.class_num).cuda()

    args.modelpath = args.output_dir + '/target_F_' + args.savename + ".pt"
    netF.load_state_dict(torch.load(args.modelpath))
    netF_c.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/target_B_' + args.savename + ".pt"
    netB.load_state_dict(torch.load(args.modelpath))
    netB_c.load_state_dict(torch.load(args.modelpath))
    # args.modelpath = args.output_dir + '/target_C_' + args.savename + ".pt"
    # netC.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/target_BRF_' + args.savename + ".pt"    
    netBRF.load_state_dict(torch.load(args.modelpath))
    netBRF_c.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/target_CRF_' + args.savename + ".pt"    
    netCRF.load_state_dict(torch.load(args.modelpath))
    netCRF_c.load_state_dict(torch.load(args.modelpath))

    netCRF_c.eval()
    for k, v in netCRF_c.named_parameters():
        v.requires_grad = False
    netF_c.eval()
    for k, v in netF_c.named_parameters():
        v.requires_grad = False 
    netB_c.eval()
    for k, v in netB_c.named_parameters():
        v.requires_grad = False   
    netBRF_c.eval()
    for k, v in netBRF_c.named_parameters():
        v.requires_grad = False  

    param_group = []

    for k, v in netC.named_parameters():
        if args.lr_decayc_2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decayc_2}]
        else:
            v.requires_grad = False        

    if args.train_step2 == 0.0:
        netCRF.eval()
        for k, v in netCRF.named_parameters():
            v.requires_grad = False
        netF.eval()
        for k, v in netF.named_parameters():
            v.requires_grad = False 
        netB.eval()
        for k, v in netB.named_parameters():
            v.requires_grad = False   
        netBRF.eval()
        for k, v in netBRF.named_parameters():
            v.requires_grad = False        
    else:
        for k, v in netCRF.named_parameters():
            if args.lr_decayo_2 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decayo_2}]
            else:
                v.requires_grad = False   
        for k, v in netF.named_parameters():
            if args.lr_decayo_2 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decayo_2}]
            else:
                v.requires_grad = False
        for k, v in netB.named_parameters():
            if args.lr_decayo_2 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decayo_2}]
            else:
                v.requires_grad = False
        for k, v in netBRF.named_parameters():
            if args.lr_decayo_2 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decayo_2}]
            else:
                v.requires_grad = False    



    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    # interval_iter = max_iter // args.interval
    interval_iter = len(dset_loaders["target"])
    iter_num = 0

    classifier_loss_total = 0.0
    classifier_loss_count = 0
    # entropy_loss_total = 0.0
    # entropy_loss_count = 0
    # costlog_loss_total = 0.0
    # costlog_loss_count = 0
    # costs_loss_total = 0.0
    # costs_loss_count = 0
    # costdist_loss_total = 0.0
    # costdist_loss_count = 0
    # right_sample_count = 0
    # sum_sample = 0
    # start_output = True

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test_1 = netB(netF(inputs_test))
        features_test_2 = netBRF(features_test_1)
        outputs_test_rf = netCRF_c(netBRF_c(netB_c(netF_c(inputs_test)))).detach()
        # outputs_test_rf = netCRF(features_test_2)
        outputs_test_h = netC(features_test_2)

        _, pred_rf = torch.max(outputs_test_rf, 1)


        # if (args.temp2 != 0.0):
        # 	outputs_test_h_softmax = nn.Softmax(dim=1)(outputs_test_h/args.temp2)
        # 	classifier_loss = nn.CrossEntropyLoss()(outputs_test_h_softmax, pred_rf)
        # else:
        # 	classifier_loss = nn.CrossEntropyLoss()(outputs_test_h, pred_rf)
        alpha = 0.5
        p = F.log_softmax(outputs_test_h/args.temp2, dim=1)
        q = F.softmax(outputs_test_rf/args.temp2, dim=1)
        l_kl = F.kl_div(p, q, size_average=False) * (args.temp2**2) / outputs_test_h.shape[0]
        l_ce = F.cross_entropy(outputs_test_h, pred_rf)
        classifier_loss = l_kl * alpha + l_ce * (1.0 - alpha)

        
        if (args.w_vat_2 > 0):
        	eps = (torch.randn(size=inputs_test.size())).type(inputs_test.type())
        	eps = 1e-6 * normalize_perturbation(eps)
        	eps.requires_grad = True
        	outputs_source_adv_eps = netC(netBRF(netB(netF(inputs_test + eps))))
        	loss_func_nll = KLDivWithLogits()
        	loss_eps  = loss_func_nll(outputs_source_adv_eps, outputs_test_h.detach())
        	loss_eps.backward()
        	eps_adv = eps.grad
        	eps_adv = normalize_perturbation(eps_adv)
        	inputs_source_adv = inputs_test + args.radius_2 * eps_adv
        	output_source_adv = netC(netBRF(netB(netF(inputs_source_adv.detach()))))
        	loss_vat     = loss_func_nll(output_source_adv, outputs_test_h.detach())
        	classifier_loss += args.w_vat_2 * loss_vat

        classifier_loss_total += classifier_loss
        classifier_loss_count += 1   
        # entropy_loss_total += entropy_loss
        # entropy_loss_count += 1 
        # costlog_loss_total += 0
        # costlog_loss_count += 1  
        # costs_loss_total += 0
        # costs_loss_count += 1  
        # costdist_loss_total += 0
        # costdist_loss_count += 1 

        # max_hyperplane = outputs_test.max(dim=1).values       
        # max_hyperplane[max_hyperplane > 0] = 1
        # max_hyperplane[max_hyperplane < 0] = 0
        # right_sample_count += max_hyperplane.sum()
        # sum_sample += outputs_test.shape[0]

        # if (start_output):
        #     all_output = outputs_test.float().cpu()
        #     start_output = False
        # else:
        #     all_output = torch.cat((all_output, outputs_test.float().cpu()), 0)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netC.eval()
            if args.train_step2 != 0.0:
            	netF.eval()
            	netB.eval()
            	netCRF.eval()
            	netBRF.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                # _, predict = torch.max(all_output, 1)
                acc_s_tr, _ = cal_acc2(dset_loaders['target'], netF, netB, netC, netBRF, False)
                acc_s_te, _ = cal_acc2(dset_loaders['test'], netF, netB, netC, netBRF, False)
                acc_s_tr_rf, _ = cal_accrf(dset_loaders['target'], netF, netB, netBRF, netCRF, False)
                acc_s_te_rf, _ = cal_accrf(dset_loaders['test'], netF, netB, netBRF, netCRF, False)
                

                # log_str = 'Task: {}, Iter:{}/{}; Loss : {:.2f}, , Accuracy target (train/test) = {:.2f}% / {:.2f}%, moved samples: {}/{}.'.format(args.name, iter_num, max_iter, \
                # classifier_loss_total/classifier_loss_count, acc_s_tr, acc_s_te, right_sample_count, sum_sample)
                log_str = 'Task: {}, Iter:{}/{}; Loss : {:.2f}, , Accuracy target (trainh/testh/trainrf/testrf) = {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}%.'.format(args.name, iter_num, max_iter, \
                classifier_loss_total/classifier_loss_count, acc_s_tr, acc_s_te, acc_s_tr_rf, acc_s_te_rf)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            # if acc_s_te < 50:
            #     return netF, netB, netC, (acc_s_te + 100)

            classifier_loss_total = 0.0
            classifier_loss_count = 0

            netC.train()
            if args.train_step2 != 0.0:
            	netF.train()
            	netB.train()
            	netCRF.train()
            	netBRF.train()

    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target2_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target2_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target2_C_" + args.savename + ".pt"))
        torch.save(netBRF.state_dict(), osp.join(args.output_dir, "target2_BRF_" + args.savename + ".pt"))
        torch.save(netCRF.state_dict(), osp.join(args.output_dir, "target2_CRF_" + args.savename + ".pt"))

    acc_s_te_rf, _ = cal_accrf(dset_loaders['test'], netF_c, netB_c, netBRF_c, netCRF_c, False)
    print("Test debug:", acc_s_te_rf)
    return netF, netB, netC, acc_s_te

def test_target2_1(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netBRF = network.feat_bootleneck_rf(nrf=args.nrf, type=args.classifier, gamma = args.gamma, bottleneck_dim=args.bottleneck).cuda()
    netCRF = network.feat_classifier_rf(nrf=args.nrf, type=args.layer_rf, class_num = args.class_num).cuda()

    args.modelpath = args.output_dir + '/target_F_' + args.savename + ".pt" 
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/target_B_' + args.savename + ".pt" 
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/target_C_' + args.savename + ".pt" 
    netC.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/target_BRF_' + args.savename + ".pt"     
    netBRF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/target_CRF_' + args.savename + ".pt"     
    netCRF.load_state_dict(torch.load(args.modelpath))

    netF.eval()
    netB.eval()
    netC.eval()
    netBRF.eval()
    netCRF.eval()

    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB, netC)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name, acc_os2, acc_os1, acc_unknown)
    else:
        if args.dset=='VISDA-C':
            acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
        else:
            acc_h, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            acc_rf, _ = cal_accrf(dset_loaders['test'], netF, netB, netBRF, netCRF, False)
            log_str = '\nTraining:, Task: {}, Accuracy (h/rf) = {:.2f}% / {:.2f}%'.format(args.name, acc_h, acc_rf)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label(loader, netF, netB, netC, args):
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
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int')

def obtain_labelrf(loader, netF, netB, netBRF, netCRF, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            feas = netB(netF(inputs))
            # outputs = netC(feas)
            outputs = netCRF(netBRF(feas))

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
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy RF = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=50, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    # parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)

    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='ckps/target/')
    parser.add_argument('--output_src', type=str, default='ckps/source/')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)

    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--gent', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--cls_parrf', type=float, default=0.3)
    parser.add_argument('--nrf', type=int, default=512)
    parser.add_argument('--max_zero', type=float, default=1.0)
    parser.add_argument('--w_vat', type=float, default=0.0)
    parser.add_argument('--alpha_rf', type=float, default=0.1)
    parser.add_argument('--alpha_rfen', type=float, default=0.1)

    parser.add_argument('--radius', type=float, default=0.01)    
    parser.add_argument('--layer_rf', type=str, default="linear", choices=["linear", "wn"])
    parser.add_argument('--grid', type=float, default=0.0)    
    parser.add_argument('--train_c', type=float, default=0.0)    
    parser.add_argument('--train_rf', type=float, default=0.0)
    parser.add_argument('--lr_decayc', type=float, default=0.1)

    parser.add_argument('--fromShot', type=float, default=0.0)

    parser.add_argument('--radius_2', type=float, default=0.01)  
    parser.add_argument('--lr_decayc_2', type=float, default=1.0)

    parser.add_argument('--train_step2', type=float, default=0.0)
    parser.add_argument('--lr_decayo_2', type=float, default=0.1)
    parser.add_argument('--temp2', type=float, default=0.0)
    parser.add_argument('--w_vat_2', type=float, default=0.0)
    parser.add_argument('--layer_2', type=str, default="wn", choices=["linear", "wn"])
   
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)
    # np.random.seed(SEED)
    # random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        if i != args.t:
        	continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        # test_target(args)
        test_target2_1(args)

        if (args.grid > 0.0):
            list_cls_par = [0.3, 0.1, 1.0]
            # list_w_vat = [0.1, 1.0, 0.0]
            list_w_vat = [0.0]
            list_alpha_rfen = [0.1, 1.0]
            list_alpha_rf = [0.1, 1.0, 0.0]
            list_max_zero = [1.0, 0.0]
            args.max_epoch = 30

            # list_cls_par = [1.0]
            # list_w_vat = [1.0]
            # list_alpha_rfen = [0.1]
            # list_alpha_rf = [0.1]
            # list_max_zero = [0.0]

        else:
            list_cls_par = [args.cls_par]
            list_w_vat = [args.w_vat]
            list_alpha_rfen = [args.alpha_rfen]
            list_alpha_rf = [args.alpha_rf]
            list_max_zero = [args.max_zero]

        dict_result = dict()
        for cur_cls_par in list_cls_par:
            args.cls_par = cur_cls_par        
            for cur_w_vat in list_w_vat:
                args.w_vat = cur_w_vat
                for cur_alpha_rfen in list_alpha_rfen:
                    args.alpha_rfen = cur_alpha_rfen                
                    for cur_alpha_rf in list_alpha_rf:
                        args.alpha_rf = cur_alpha_rf
                        for cur_max_zero in list_max_zero:
                            args.max_zero = cur_max_zero
                            torch.manual_seed(SEED)
                            torch.cuda.manual_seed(SEED)
                            np.random.seed(SEED)
                            random.seed(SEED)
                            
                            # _,_,_, acc = train_target(args)
                            # dict_result[(args.cls_par, args.alpha_rfen, args.alpha_rf, args.max_zero, args.w_vat)] = acc
                            # for key in dict_result:
                            #     print("{}-{}-{}-{}-{}-{}".format(key[0], key[1], key[2], key[3], key[4], dict_result[key]))
                            _,_,_, acc = train_target2(args)
                            dict_result[(args.lr_decayc_2, args.train_step2, args.lr_decayo_2, args.temp2, args.w_vat_2, args.layer_2)] = acc
                            for key in dict_result:
                                print("{}-{}-{}-{}-{}-{}-{}".format(key[0], key[1], key[2], key[3], key[4], key[5], dict_result[key]))