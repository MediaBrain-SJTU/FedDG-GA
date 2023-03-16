import os
import argparse
import torch
from network.get_network import GetNetwork
from utils.log_utils import *
from torch.utils.tensorboard.writer import SummaryWriter
from data.pacs_dataset import PACS_FedDG
from utils.classification_metric import Classification 
from utils.fed_merge import FedAvg, FedUpdate
from utils.trainval_func import site_evaluation, SaveCheckPoint
from utils.weight_adjust import refine_weight_dict_by_GA
from network.FedOptimizer.Scaffold import *
import torch.nn.functional as F
from tqdm import tqdm

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='pacs', choices=['pacs'], help='Name of dataset')
    parser.add_argument("--model", type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'], help='model name')
    parser.add_argument("--test_domain", type=str, default='p',
                        choices=['p', 'a', 'c', 's'], help='the domain name for testing')
    parser.add_argument('--num_classes', help='number of classes default 7', type=int, default=7)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=16)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=5)
    parser.add_argument('--comm', help='epochs number', type=int, default=40)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--step_size', help='rate weight step', type=float, default=0.2)
    parser.add_argument("--lr_policy", type=str, default='step', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--fair", type=str, default='acc', choices=['acc', 'loss'],
                        help="the fairness metric for FedAvg")
    parser.add_argument('--note', help='note of experimental settings', type=str, default='generalization_adjustment')
    parser.add_argument('--display', help='display in controller', action='store_true') 

    return parser.parse_args()

def epoch_site_train(epochs, site_name, model, optimzier, scheduler, c_ci, dataloader, log_ten, metric):
    model.train()
    for i, data_list in enumerate(dataloader):
        imgs, labels, domain_labels = data_list
        imgs = imgs.cuda()
        labels = labels.cuda()
        domain_labels = domain_labels.cuda()
        optimzier.zero_grad()
        output = model(imgs)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimzier.step(c_ci) 
        log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epochs*len(dataloader)+i)
        metric.update(output, labels)
    
    log_ten.add_scalar(f'{site_name}_train_acc', metric.results()['acc'], epochs)
    scheduler.step()
    
def site_train(comm_rounds, site_name, args, model, optimizer, scheduler, c_ci, dataloader, log_ten, metric):
    tbar = tqdm(range(args.local_epochs))
    for local_epoch in tbar:
        tbar.set_description(f'{site_name}_train')
        epoch_site_train(comm_rounds*args.local_epochs + local_epoch, site_name, model, optimizer, scheduler, c_ci,  dataloader, log_ten, metric)


def GetFedModel(args, num_classes, is_train=True):
    global_model, feature_level = GetNetwork(args, args.num_classes, True)
    global_model = global_model.cuda()
    model_dict = {}
    optimizer_dict = {}
    scheduler_dict = {}
    ci_dict = {}
    c = GenZeroParamList(global_model)
    
    for domain_name in ['p', 'a', 'c', 's']:
        model_dict[domain_name], _ = GetNetwork(args, num_classes, is_train)
        model_dict[domain_name] = model_dict[domain_name].cuda()
        
        ci_dict[domain_name] = GenZeroParamList(model_dict[domain_name])
        
        optimizer_dict[domain_name] = Scaffold(model_dict[domain_name].parameters(), lr=args.lr, momentum=0.9,
                                                      weight_decay=5e-4)
        if args.lr_policy == 'step':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.StepLR(optimizer_dict[domain_name], step_size=args.local_epochs * args.comm, gamma=0.1)
    return global_model, model_dict, optimizer_dict, scheduler_dict, ci_dict, c

   

def main():
    file_name = 'GA_'+os.path.split(__file__)[1].replace('.py', '')
    args = get_argparse()
    log_dir, tensorboard_dir = Gen_Log_Dir(args, file_name=file_name)
    log_ten = SummaryWriter(log_dir=tensorboard_dir)
    log_file = Get_Logger(file_name=log_dir + 'train.log', display=args.display)
    Save_Hyperparameter(log_dir, args)
    
    '''dataset and dataloader'''
    dataobj = PACS_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    dataloader_dict, dataset_dict = dataobj.GetData()
    
    '''model相关'''
    metric = Classification()
    # 模型定义
    global_model, model_dict, optimizer_dict, scheduler_dict, ci_dict, c = GetFedModel(args, args.num_classes)
    '''weight_dict相关'''
    weight_dict = {}
    site_results_before_avg = {}
    site_results_after_avg = {}
    # 强行保持一致
    for site_name in dataobj.train_domain_list:
        weight_dict[site_name] = 1./3.
        site_results_before_avg[site_name] = None
        site_results_after_avg[site_name] = None
        
    FedUpdate(model_dict, global_model)
    step_size_decay = args.step_size / args.comm
    best_val = 0.
    for i in range(args.comm+1):
        FedUpdate(model_dict, global_model)
        for domain_name in dataobj.train_domain_list:
            c_ci = ListMinus(c, ci_dict[domain_name])  
            K = len(dataloader_dict[domain_name]['train']) * args.local_epochs
            site_train(i, domain_name, args, model_dict[domain_name], optimizer_dict[domain_name], 
                       scheduler_dict[domain_name], c_ci, dataloader_dict[domain_name]['train'], log_ten, metric)
            
            site_results_before_avg[domain_name] = site_evaluation(i, domain_name, args, model_dict[domain_name], dataloader_dict[domain_name]['val'], log_file, log_ten, metric, note='before_fed')

            ci_dict[domain_name] = UpdateLocalControl(c, ci_dict[domain_name], global_model,  model_dict[domain_name], K)

        c = UpdateServerControl(c,ci_dict, weight_dict)        
        FedAvg(model_dict, weight_dict, global_model)
        
        fed_val = 0.
        for domain_name in dataobj.train_domain_list:
            site_results_after_avg[domain_name] = site_evaluation(i, domain_name, args, global_model, dataloader_dict[domain_name]['val'], log_file, log_ten, metric)
            fed_val+= site_results_after_avg[domain_name]['acc']*weight_dict[domain_name]

        if fed_val >= best_val:
            best_val = fed_val
            SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='best_val_model')
            for domain_name in dataobj.train_domain_list:
                SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'best_val_{domain_name}_model')
            log_file.info(f'Model saved! Best Val Acc: {best_val*100:.2f}%')
        site_evaluation(i, args.test_domain, args, global_model, dataloader_dict[args.test_domain]['test'], log_file, log_ten, metric, note='test_domain')
        weight_dict = refine_weight_dict_by_GA(weight_dict, site_results_before_avg, site_results_after_avg, args.step_size - (i-1)*step_size_decay, fair_metric=args.fair)
        log_str = f'Round {i} FedAvg weight: {weight_dict}'
        log_file.info(log_str)
        
    SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='last_model')
    for domain_name in dataobj.train_domain_list:
        SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'last_{domain_name}_model')
    
    
if __name__ == '__main__':
    main()