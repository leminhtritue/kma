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
from loss import CrossEntropyLabelSmooth

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

def cal_accrf(loader, netF, netB, netBRF, netCRF):
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

def train_source(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netBRF = network.feat_bootleneck_rf(nrf=args.nrf, type=args.classifier, gamma = args.gamma, bottleneck_dim=args.bottleneck).cuda()
    netCRF = network.feat_classifier_rf(nrf=args.nrf, type=args.layer_rf, class_num = args.class_num).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   

    for k, v in netCRF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netBRF.named_parameters():
        if (args.train_brf != 0.0):
            param_group += [{'params': v, 'lr': learning_rate}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    netCRF.train()
    if (args.train_brf != 0.0):
        netBRF.train()

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
        output_latent = netB(netF(inputs_source))
        outputs_source = netC(output_latent)
        outputs_source_rf = netCRF(netBRF(output_latent))

        classifier_loss = args.w_ce_h * CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source) 

        if (args.kernel_softmax == 0.0):
            classifier_loss += args.w_kernel * loss.KernelSource(num_classes=args.class_num, alpha=args.w_kernel_w_reg)(outputs_source_rf, labels_source, netCRF)
        else:
            classifier_loss += args.w_kernel * CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source_rf, labels_source)

        if (args.w_vat > 0):
            eps = (torch.randn(size=inputs_source.size())).type(inputs_source.type())
            eps = 1e-6 * normalize_perturbation(eps)
            eps.requires_grad = True
            outputs_source_adv_eps = netC(netB(netF(inputs_source + eps)))
            loss_func_nll = KLDivWithLogits()
            loss_eps  = loss_func_nll(outputs_source_adv_eps, outputs_source.detach())
            loss_eps.backward()
            eps_adv = eps.grad
            eps_adv = normalize_perturbation(eps_adv)
            inputs_source_adv = inputs_source + args.radius * eps_adv
            output_source_adv = netC(netB(netF(inputs_source_adv.detach())))
            loss_vat     = loss_func_nll(output_source_adv, outputs_source.detach())

            classifier_loss += args.w_vat * loss_vat

        if (args.w_vat_rf > 0):
            eps_rf = (torch.randn(size=inputs_source.size())).type(inputs_source.type())
            eps_rf = 1e-6 * normalize_perturbation(eps_rf)
            eps_rf.requires_grad = True
            outputs_source_adv_eps_rf = netCRF(netBRF(netB(netF(inputs_source + eps_rf))))
            loss_func_nll_rf = KLDivWithLogits()
            loss_eps_rf  = loss_func_nll_rf(outputs_source_adv_eps_rf, outputs_source_rf.detach())
            loss_eps_rf.backward()
            eps_adv_rf = eps_rf.grad
            eps_adv_rf = normalize_perturbation(eps_adv_rf)
            inputs_source_adv_rf = inputs_source + args.radius * eps_adv_rf
            output_source_adv_rf = netCRF(netBRF(netB(netF(inputs_source_adv_rf.detach()))))
            loss_vat_rf     = loss_func_nll_rf(output_source_adv_rf, outputs_source_rf.detach())

            classifier_loss += args.w_vat_rf * loss_vat_rf

        total_loss += classifier_loss
        count_loss += 1   

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            netCRF.eval()
            if (args.train_brf != 0.0):
                netBRF.eval()

            acc_s_tr, _ = cal_acc(dset_loaders['source_tr'], netF, netB, netC)
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC)
            acc_s_tgt, _ = cal_acc(dset_loaders['test'], netF, netB, netC)

            log_str = 'Task: {}, Iter:{}/{}; Accuracy source (train / test / target) = {:.2f}% / {:.2f}%'.format(args.dset, iter_num, max_iter, acc_s_tr, acc_s_te, acc_s_tgt)
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
                best_netBRF = netBRF.state_dict()
                best_netCRF = netCRF.state_dict()
            
            netF.train()
            netB.train()
            netC.train()
            netCRF.train()
            if (args.train_brf != 0.0):
                netBRF.train()

    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))
    torch.save(best_netBRF, osp.join(args.output_dir, "source_BRF.pt"))
    torch.save(best_netCRF, osp.join(args.output_dir, "source_CRF.pt"))
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

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netBRF = network.feat_bootleneck_rf(nrf=args.nrf, type=args.classifier, gamma = args.gamma, bottleneck_dim=args.bottleneck).cuda()
    netCRF = network.feat_classifier_rf(nrf=args.nrf, type=args.layer_rf, class_num = args.class_num).cuda()

    args.modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_BRF.pt'
    netBRF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_CRF.pt'
    netCRF.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()
    netBRF.eval()
    netCRF.eval()

    acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
    accrf, _ = cal_accrf(dset_loaders['test'], netF, netB, netBRF, netCRF)
    log_str = 'Task: {}, Accuracy trained source (H / RF)= {:.2f}% / {:.2f}%'.format(args.dset, acc, accrf)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

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

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netBRF = network.feat_bootleneck_rf(nrf=args.nrf, type=args.classifier, gamma = args.gamma, bottleneck_dim=args.bottleneck).cuda()
    netCRF = network.feat_classifier_rf(nrf=args.nrf, type=args.layer_rf, class_num = args.class_num).cuda()

    args.modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_BRF.pt'
    netBRF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_CRF.pt'
    netCRF.load_state_dict(torch.load(args.modelpath))

    netC.eval()
    netCRF.eval()
    netBRF.eval()

    for k, v in netC.named_parameters():
        v.requires_grad = False
    for k, v in netCRF.named_parameters():
        v.requires_grad = False
    for k, v in netBRF.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = len(dset_loaders["target"])
    # interval_iter = max_iter // args.interval
    iter_num = 0

    classifier_loss_total = 0.0
    classifier_loss_count = 0

    while iter_num < max_iter:
        optimizer.zero_grad()
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
            mem_label = obtain_label(dset_loaders['target_te'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        if iter_num % interval_iter == 0 and args.cls_parrf > 0:
            netF.eval()
            netB.eval()
            mem_label_rf = obtain_labelrf(dset_loaders['target_te'], netF, netB, netBRF, netCRF, args)
            mem_label_rf = torch.from_numpy(mem_label_rf).cuda()
            netF.train()
            netB.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_test = inputs_test.cuda()
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        outputs_test_rf = netCRF(netBRF(features_test))

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_test, pred)
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.cls_parrf > 0:
            pred_rf = mem_label_rf[tar_idx]
            classifier_loss_rf = args.cls_parrf * nn.CrossEntropyLoss()(outputs_test_rf, pred_rf)
            classifier_loss += classifier_loss_rf

        softmax_out = nn.Softmax(dim=1)(outputs_test)
        if args.ent:
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        softmax_si = nn.Softmax(dim=1)(outputs_test_rf)
        if args.alpha_rf > 0:
            # entropy_si = -(softmax_out * torch.log(softmax_si + 1e-5))
            entropy_si = -(softmax_out * torch.log(softmax_si.detach() + args.epsilon))
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

        classifier_loss_total += classifier_loss
        classifier_loss_count += 1  

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            acc_tr, _ = cal_acc(dset_loaders['target'], netF, netB, netC)
            acc_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
            acc_tr_rf, _ = cal_accrf(dset_loaders['target'], netF, netB, netBRF, netCRF)
            acc_te_rf, _ = cal_accrf(dset_loaders['test'], netF, netB, netBRF, netCRF)

            log_str = 'Task: {}, Iter:{}/{}; Loss : {:.2f}; Accuracy target (train / test / trainrf / testrf) = {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}%.'.format(args.dset, iter_num, max_iter, \
                classifier_loss_total/classifier_loss_count, acc_tr, acc_te, acc_tr_rf, acc_te_rf)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            classifier_loss_total = 0.0
            classifier_loss_count = 0
            netF.train()
            netB.train()

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        torch.save(netBRF.state_dict(), osp.join(args.output_dir, "target_BRF_" + args.savename + ".pt"))
        torch.save(netCRF.state_dict(), osp.join(args.output_dir, "target_CRF_" + args.savename + ".pt"))

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

def obtain_labelrf(loader, netF, netB, netBRF, netCRF, args, c=None):
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
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='s2m', choices=['u2m', 'm2u','s2m'])
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--issave', type=bool, default=False)


    parser.add_argument('--epsilon', type=float, default=1e-5)

    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--nrf', type=int, default=512)
    parser.add_argument('--layer_rf', type=str, default="wn", choices=["linear", "wn"])

    parser.add_argument('--w_kernel', type=float, default=0.1)
    parser.add_argument('--w_kernel_w_reg', type=float, default=0.1)
    parser.add_argument('--w_ce_h', type=float, default=1.0)  

    parser.add_argument('--w_vat', type=float, default=0.0)
    parser.add_argument('--w_vat_rf', type=float, default=0.0)
    parser.add_argument('--radius', type=float, default=0.01)

    parser.add_argument('--train_brf', type=float, default=0.0)
    parser.add_argument('--kernel_softmax', type=float, default=0.0)

    parser.add_argument('--cls_parrf', type=float, default=0.0)
    parser.add_argument('--alpha_rf', type=float, default=0.0)
    parser.add_argument('--alpha_rfen', type=float, default=0.0)

    args = parser.parse_args()
    args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = str((int(args.gpu_id) % 4))
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
