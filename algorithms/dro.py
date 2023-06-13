from .model import InferEnv, InferEnvMultiClass
import pdb
import torch
from torch.nn import Parameter
import pandas as pd
import numpy as np
import wandb

class DRO:

    def __init__(self, flags, dp):
        eta_init =torch.tensor([13.0],requires_grad=True).cuda()
        self.eta = torch.nn.parameter.Parameter(eta_init)
        # print("eta_init",eta_init)
        self.lamda=torch.nn.parameter.Parameter(torch.tensor([100.0],requires_grad=True).cuda())
        # print("lamda",self.lamda)
        self.infer_env_mean=torch.nn.parameter.Parameter(torch.tensor([0.5],requires_grad=True).cuda())
        # print("infer_env_mean",self.infer_env_mean)
        if flags.dataset == "logit_z":
            self.infer_env = InferEnv(flags, z_dim=1).cuda()
        elif flags.dataset == "celebaz_feature":
            self.infer_env = InferEnvMultiClass(flags, z_dim=flags.aux_num, class_num=flags.z_class_num).cuda()
        elif flags.dataset == "house_price":
            self.infer_env = InferEnvMultiClass(flags, z_dim=1, class_num=flags.z_class_num).cuda()
        elif flags.dataset == 'landcover':
            self.infer_env = InferEnvMultiClass(flags, z_dim=flags.aux_num, class_num=flags.z_class_num).cuda()
        else:
            raise Exception
        self.params=list(self.infer_env.parameters()) # +[self.eta,]
        self.optimizer_infer_env = torch.optim.Adam(self.params, lr=0.01)
        self.flags = flags
        
        # self.relu=torch.nn.ReLU()
        self.relu=torch.nn.LeakyReLU(0.2)

    def __call__(self, batch_data, step,  mlp=None, scale=None, **kwargs):
        train_x, train_y, train_z, train_g, train_c, train_invnoise = batch_data
        normed_z = (train_z.float() - train_z.float().mean()) /train_z.float().std()
        normed_x = (train_x.float() - train_x.float().mean()) /train_x.float().std()
        
        train_logits = scale * mlp(train_x)
        if self.flags.dataset == "house_price":
            loss_fun = torch.nn.MSELoss(reduction='none')
            train_nll = loss_fun(train_logits, train_y)
        else:
            train_nll = torch.nn.functional.binary_cross_entropy_with_logits(
                train_logits, train_y, reduction="none")
       

        infered_envs = self.infer_env(normed_x)

        cons_infered_envs=(infered_envs.mean()-self.infer_env_mean)**2

        train_loss = (infered_envs*self.relu(train_nll-self.eta)+self.eta).mean()-cons_infered_envs*self.lamda

        if step < self.flags.penalty_anneal_iters:
            # gradient ascend on infer_env net
            
            self.optimizer_infer_env.zero_grad()
            (-train_loss).backward(retain_graph=True,inputs=list(self.params))
            self.optimizer_infer_env.step()

        train_penalty = torch.tensor(0.0)
        
        #save infered envs to csv
        torch.save(self.infer_env.state_dict(), "infered_envs1.pt")
        
        # print(self.eta)
        
        return train_loss, train_penalty


