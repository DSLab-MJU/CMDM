from abc import abstractmethod

import math
import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torchvision import datasets, transforms, utils, models
from torchvision.utils import save_image, make_grid
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Subset
from torch.utils import data

from modules import *
from U_Net_mask import *

def ddpm_schedules(beta1, beta2, T):
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  
        "oneover_sqrta": oneover_sqrta,  
        "sqrt_beta_t": sqrt_beta_t, 
        "alphabar_t": alphabar_t,  
        "sqrtab": sqrtab, 
        "sqrtmab": sqrtmab,  
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  
    }


class Diffusion(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.0):
        super().__init__()
        self.nn_model = nn_model.to(device)
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, conds):
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  
        drop_cond = torch.ones(conds.shape[0],1)
        probability_conds = torch.bernoulli(torch.zeros_like(drop_cond)+self.drop_prob) #0.1의 확률로 0으로 만들어서
        for i in range(drop_cond.shape[0]):
            if probability_conds[i] == 1:
                conds[i] = 0
        return self.loss_mse(noise, self.nn_model(x_t, _ts / self.n_T, conds))

    def sample(self, n_sample, size, device, condition, guide_w = 0.0):

        x_i = torch.randn(n_sample, *size).to(device)  

        if not isinstance(condition, torch.Tensor):
            condition = torch.tensor(condition)
        c_i = condition.to(device) if condition.device != device else condition 
        context_conds = torch.zeros_like(c_i).to(device)
        context_conds = torch.cat([c_i, context_conds], dim=0)
        context_conds = context_conds.type(torch.float)
        for i in range(self.n_T, 0, -1): 
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,)
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.nn_model(x_i, t_is, context_conds)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        return x_i
