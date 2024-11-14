#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
import os
import shutil
import sys
import warnings
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import pdb

from helpers.datasets import partition_data
from helpers.synthesizers import AdvSynthesizer
from helpers.utils import get_dataset, average_weights, DatasetSplit, KLDiv, setup_seed, test
from models.generator import Generator
from models.nets import CNNCifar, CNNMnist, CNNCifar100
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F

from models.resnet import resnet18
from models.vit import deit_tiny_patch16_224
import wandb

warnings.filterwarnings('ignore')
upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)



class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.dataset = dataset
        self.args = args
        self.train_loader = DataLoader(DatasetSplit(dataset, idxs),
                                       batch_size=self.args.local_bs, shuffle=True, num_workers=self.args.num_workers)

    def update_weights(self, model, client_id):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=self.args.momentum)
        
        if self.args.LDP:
            ## opacus
            # from from opacus.validators import ModuleValidator
            # from opacus import PrivacyEngine
            # errors = ModuleValidator.validate(model, strict=False)
            # print(errors)
            
            # model, optimizer, self.train_loader = privacy_engine.make_private(
            #                                                             module=model,
            #                                                             optimizer=optimizer,
            #                                                             data_loader=self.train,
            #                                                             max_grad_norm=1.0,
            #                                                             noise_multiplier=1.0,
            #                                                         )
            
            ## torchdp 
            # from torchdp import PrivacyEngine
            # privacy_engine = PrivacyEngine(
            #     model,
            #     args.batch_size,
            #     len(self.train_loader.dataset),
            #     alphas=[1, 10, 100],
            #     noise_multiplier=1.3,
            #     max_grad_norm=1.0,
            # )
            # privacy_engine.attach(optimizer)
            
            
            # pyvacy as in https://github.com/ChrisWaites/pyvacy
            
            from pyvacy import optim, analysis, sampling
            
            training_parameters = {
                'N': len(self.dataset),
                
                
                # An upper bound on the L2 norm of each gradient update.
                # A good rule of thumb is to use the median of the L2 norms observed
                # throughout a non-private training loop.
                'l2_norm_clip': self.args.l2_norm_clip,
                # A coefficient used to scale the standard deviation of the noise applied to gradients.
                'noise_multiplier': self.args.noise_multiplier,
                # Each example is given probability of being selected with minibatch_size / N.
                # Hence this value is only the expected size of each minibatch, not the actual. 
                'minibatch_size': self.args.minibatch_size,
                # Each minibatch is partitioned into distinct groups of this size.
                # The smaller this value, the less noise that needs to be applied to achieve
                # the same privacy, and likely faster convergence. Although this will increase the runtime.
                'microbatch_size': self.args.microbatch_size,
                # The usual privacy parameter for (ε,δ)-Differential Privacy.
                # A generic selection for this value is 1/(N^1.1), but it's very application dependent.
                'delta': self.args.delta,
                # The number of minibatches to process in the training loop.
                'iterations': self.args.iterations,
                
                
                # added !
                'lr': self.args.lr,
                'momentum': self.args.momentum
                
            }
            print("training params: ", training_parameters)
            # model ok
            optimizer = optim.DPSGD(params=model.parameters(), 
                                    l2_norm_clip=training_parameters['l2_norm_clip'],
                                    noise_multiplier=training_parameters['noise_multiplier'],
                                    minibatch_size=training_parameters['minibatch_size'],
                                    microbatch_size=training_parameters['microbatch_size'],
                                    lr=self.args.lr,
                                    momentum=0.9)
            epsilon = analysis.epsilon(N=training_parameters['N'],
                                       batch_size=training_parameters['microbatch_size'],
                                       noise_multiplier=training_parameters['noise_multiplier'],
                                       iterations=training_parameters['iterations'],
                                       delta=training_parameters['delta'],
                                       )
            print("###########EPSILON################", epsilon)
            
            # loaders for the data (functions)
            
            minibatch_loader, microbatch_loader = sampling.get_data_loaders(microbatch_size=training_parameters['microbatch_size'],
                                                                            minibatch_size=training_parameters['minibatch_size'],
                                                                            iterations=training_parameters['iterations'],
                                                                            )
            
            

        # label_list = [0] * 100
        # for batch_idx, (images, labels) in enumerate(self.train_loader):
        #     for i in range(100):
        #         label_list[i] += torch.sum(labels == i).item()
        # print(label_list)
        local_acc_list = []
        
        ##DP
        if self.args.LDP:
            for X_minibatch, y_minibatch in tqdm(minibatch_loader(self.dataset)):
                optimizer.zero_grad()
                for X_microbatch, y_microbatch in tqdm(microbatch_loader(TensorDataset(X_minibatch, y_minibatch))):
                    X_microbatch, y_microbatch = X_microbatch.cuda(), y_microbatch.cuda()
                    optimizer.zero_microbatch_grad()
                    loss = F.cross_entropy(model(X_microbatch), y_microbatch)
                    loss.backward()
                    optimizer.microbatch_step()
                optimizer.step()

                # common part 
                acc, test_loss = test(model, test_loader)
                # if client_id == 0:
                #     wandb.log({'local_epoch': iter})
                # wandb.log({'client_{}_accuracy'.format(client_id): acc})
                local_acc_list.append(acc)
                
        ## NON DP 
        else:
            for iter in tqdm(range(self.args.local_ep)):
                
                for batch_idx, (images, labels) in enumerate(self.train_loader):
                    images, labels = images.cuda(), labels.cuda()
                    model.zero_grad()
                    # ---------------------------------------
                    output = model(images)
                    loss = F.cross_entropy(output, labels)
                    # ---------------------------------------
                    loss.backward()
                    optimizer.step()
                
                # common part 
                acc, test_loss = test(model, test_loader)
                # if client_id == 0:
                #     wandb.log({'local_epoch': iter})
                # wandb.log({'client_{}_accuracy'.format(client_id): acc})
                local_acc_list.append(acc)
                
                
        return model.state_dict(), np.array(local_acc_list)


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=100,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.') # for the dataset, not the the partition

    # Data Free
    parser.add_argument('--adv', default=0, type=float, help='scaling factor for adv loss')

    parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
    parser.add_argument('--save_dir', default='run/synthesis', type=str, help='saving directory for the data pool ? ')
    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--beta', default=0.5, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--g_steps', default=20, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers in dataloader, reduce to run on less RAM')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    # Misc
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--type', default="pretrain", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--model', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--other', default="", type=str,
                        help='seed for initializing training.')
    # Local Differential Privacy 
    parser.add_argument('--LDP', default="False", type=str, 
                        help='Whether to apply local differential privacy to the local models') # will be converted to a bool lower 
    parser.add_argument('--l2_norm_clip',
                        type=float,
                        default=1.0,
                        help='''An upper bound on the L2 norm of each gradient update.
                                A good rule of thumb is to use the median of the L2 norms observed
                                throughout a non-private training loop.
                        '''
                        )
    parser.add_argument('--noise_multiplier',
                        type=float,
                        default=1.1,
                        help='''A coefficient used to scale the standard deviation of the noise applied to gradients.
                        '''
                        )
    parser.add_argument('--minibatch_size',
                        type=int,
                        default=1000,
                        help='''Each example is given probability of being selected with minibatch_size / N.
                                Hence this value is only the expected size of each minibatch, not the actual.
                        '''
                        )
    parser.add_argument('--microbatch_size',
                        type=int,
                        default=20,
                        help='''Each minibatch is partitioned into distinct groups of this size.
                                The smaller this value, the less noise that needs to be applied to achieve
                                the same privacy, and likely faster convergence. Although this will increase the runtime.
                        '''
                        )
    parser.add_argument('--delta',
                        type=float,
                        default=1e-5,
                        help='''The usual privacy parameter for (ε,δ)-Differential Privacy.
                                A generic selection for this value is 1/(N^1.1), but it's very application dependent.
                        '''
                        )
    parser.add_argument('--iterations',
                        type=int,
                        default=400,
                        help='''
                        The number of minibatches to process in the training loop.
                        '''
                        )
    
    
    # identifier name of the run 
    parser.add_argument('--run_name', 
                        type=str,
                        default=None,
                        help='an identifier to find the run later')
        
    # parse 
    args = parser.parse_args()
    
    #  modify special args
    if args.LDP == "True":
        args.LDP = True
    elif args.LDP == "False":
        args.LDP = False 
    else:
        raise AssertionError(f"LDP is not properly assigned in line args: {args.LDP}")
    
    # debug 
    print(args, file=sys.stderr)
    
    return args


class Ensemble(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e

# class RunName():
#     # class to name the run given args, the wandb id and a user-defined tag
#     def __init__(self, args) -> None:
#         self.args = args
#         self.id = wandb.run.id
#         print("run wandb id = ", self.id)
        
#     def get_run_name(self) -> str:
#         if self.args.LDP:
#             run_name = '{}_beta{}_nbusers{}_LDP{}_its{}_clip{}_noise{}_mini{}_micro{}_delta{}'
#             run_name = run_name.format(self.args.dataset, 
#                                         self.args.beta, 
#                                         self.args.num_users, 
#                                         self.args.LDP, 
#                                         self.args.iterations, 
#                                         self.args.l2_norm_clip,
#                                         self.args.noise_multiplier,
#                                         self.args.minibatch_size,
#                                         self.args.microbatch_size,
#                                         self.args.delta)
#         else:
#             run_name = '{}_beta{}_nbusers{}_LDP{}'
#             run_name = run_name.format(self.args.dataset, 
#                                         self.args.beta, 
#                                         self.args.num_users, 
#                                         self.args.LDP)
#         return run_name
        
def kd_train(synthesizer, model, criterion, optimizer):
    student, teacher = model
    student.train()
    teacher.eval()
    description = "loss={:.4f} acc={:.2f}%"
    total_loss = 0.0
    correct = 0.0
    with tqdm(synthesizer.get_data()) as epochs:
        for idx, (images) in enumerate(epochs):
            optimizer.zero_grad()
            images = images.cuda()
            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())

            loss_s.backward()
            optimizer.step()

            total_loss += loss_s.detach().item()
            avg_loss = total_loss / (idx + 1)
            pred = s_out.argmax(dim=1)
            target = t_out.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(synthesizer.data_loader.dataset) * 100

            epochs.set_description(description.format(avg_loss, acc))


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


def get_model(args):
    if args.model == "mnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "fmnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "cnn":
        global_model = CNNCifar().cuda()
    elif args.model == "svhn_cnn":
        global_model = CNNCifar().cuda()
    elif args.model == "cifar100_cnn":
        global_model = CNNCifar100().cuda()
    elif args.model == "res":
        # global_model = resnet18()
        global_model = resnet18(num_classes=100).cuda()

    elif args.model == "vit":
        global_model = deit_tiny_patch16_224(num_classes=1000,
                                             drop_rate=0.,
                                             drop_path_rate=0.1)
        global_model.head = torch.nn.Linear(global_model.head.in_features, 10)
        global_model = global_model.cuda()
        global_model = torch.nn.DataParallel(global_model)
    return global_model


if __name__ == '__main__':
    # init 
    args = args_parser()
    
    
    wandb.init(config=args,
               project="ont-shot FL")


    setup_seed(args.seed)
    # pdb.set_trace()
    
    # run name 
    # run_name = RunName(args).get_run_name()
    run_name = args.run_name
    print("run_name : ", run_name)
    
    # Load Data
    train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
        args.dataset, args.partition, beta=args.beta, num_users=args.num_users)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)
    # BUILD MODEL

    global_model = get_model(args)
    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    local_weights = []
    global_model.train()
    acc_list = []
    users = []
    if args.type == "pretrain":
        # ===============================================
        for idx in range(args.num_users):
            print("client {}".format(idx))
            users.append("client_{}".format(idx))
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w, local_acc = local_model.update_weights(copy.deepcopy(global_model), idx)

            acc_list.append(local_acc)
            local_weights.append(copy.deepcopy(w))

        # wandb 

        for i in range(len(acc_list[0])):
            wandb.log({"client_{}_acc".format(users[c]):acc_list[c][i] for c in range(args.num_users)})
        # np.save("client_{}_acc.npy".format(args.num_users), acc_list)
        wandb.log({"client_accuracy" : wandb.plot.line_series(
            xs=[ i for i in range(len(acc_list[0])) ],
            ys=[ [acc_list[i]] for i in range(args.num_users) ],
            keys=users,
            title="Client Accuacy")})
        # torch.save(local_weights, '{}_{}.pkl'.format(name, iid))
        torch.save(local_weights, f'weights/{run_name}.pkl')
        # update global weights by FedAvg
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        print("avg acc:")
        test_acc, test_loss = test(global_model, test_loader)
        model_list = []
        for i in range(len(local_weights)):
            net = copy.deepcopy(global_model)
            net.load_state_dict(local_weights[i])
            model_list.append(net)
        ensemble_model = Ensemble(model_list)
        print("ensemble acc:")
        test(ensemble_model, test_loader)
        # ===============================================
    else:
        # ===============================================
        local_weights = torch.load(f'weights/{run_name}.pkl')
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        print("avg acc:")
        test_acc, test_loss = test(global_model, test_loader)
        model_list = []
        for i in range(len(local_weights)):
            net = copy.deepcopy(global_model)
            net.load_state_dict(local_weights[i])
            model_list.append(net)
        ensemble_model = Ensemble(model_list)
        print("ensemble acc:")
        test(ensemble_model, test_loader)
        # ===============================================
        global_model = get_model(args)
        # ===============================================

        # data generator
        nz = args.nz
        nc = 3 if "cifar" in args.dataset or args.dataset == "svhn" else 1
        img_size = 32 if "cifar" in args.dataset or args.dataset == "svhn" else 28
        generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda()
        args.cur_ep = 0
        img_size2 = (3, 32, 32) if "cifar" in args.dataset or args.dataset == "svhn" else (1, 28, 28)
        num_class = 100 if args.dataset == "cifar100" else 10
        synthesizer = AdvSynthesizer(ensemble_model, model_list, global_model, generator,
                                     nz=nz, num_classes=num_class, img_size=img_size2,
                                     iterations=args.g_steps, lr_g=args.lr_g,
                                     synthesis_batch_size=args.synthesis_batch_size,
                                     sample_batch_size=args.batch_size,
                                     adv=args.adv, bn=args.bn, oh=args.oh,
                                     save_dir=args.save_dir, dataset=args.dataset)
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        criterion = KLDiv(T=args.T)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.9)
        global_model.train()
        distill_acc = []
        for epoch in tqdm(range(args.epochs)):
            # 1. Data synthesis
            synthesizer.gen_data(args.cur_ep)  # g_steps
            args.cur_ep += 1
            kd_train(synthesizer, [global_model, ensemble_model], criterion, optimizer)  # # kd_steps
            acc, test_loss = test(global_model, test_loader)
            distill_acc.append(acc)
            is_best = acc > bst_acc
            bst_acc = max(acc, bst_acc)
            _best_ckpt = 'df_ckpt/{}.pth'.format(args.other)
            print("best acc:{}".format(bst_acc))
            save_checkpoint({
                'state_dict': global_model.state_dict(),
                'best_acc': float(bst_acc),
            }, is_best, _best_ckpt)
            wandb.log({'accuracy': acc})
        wandb.log({"global_accuracy" : wandb.plot.line_series(
            xs=[ i for i in range(args.epochs) ],
            ys=[distill_acc],
            keys=["DENSE"],
            title="Accuracy of DENSE")})
        # np.save("distill_acc_{}.npy".format(args.dataset), np.array(distill_acc)) # save accuracy

        # ===============================================


