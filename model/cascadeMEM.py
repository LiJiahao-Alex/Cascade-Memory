import datetime
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn.parameter import Parameter
from tqdm import tqdm

from tool.plugin import output_auc
from tool.trainTrick import EarlyStopping


def calculate_sim_matrix(a, b):
    abcostheta = F.linear(a, b)
    norm = F.linear(torch.norm(a, p=2, dim=1, keepdim=True), torch.norm(b, p=2, dim=1, keepdim=True))
    return abcostheta * (1 / norm)


class basicUNIT(nn.Module):
    def __init__(self, mem_size, mem_dim, nu, k):
        super().__init__()
        self.module_state = 'training'
        self.mem_flag = 'basicUNIT'
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.k = k
        self.nu = nu
        self.mem_matrix = Parameter(torch.Tensor(self.mem_size, self.mem_dim))
        stdv = 1. / math.sqrt(mem_dim)
        self.mem_matrix.data.uniform_(-stdv, stdv)

    def forward(self, encoder_outs):
        sim_matrix = calculate_sim_matrix(encoder_outs, self.mem_matrix)
        if self.module_state == 'training':
            forget_rate = self.nu
            mask_list = random.sample(range(self.mem_size), int(self.mem_size * forget_rate))
            sim_matrix[:, mask_list] *= 0
        S, I = torch.topk(sim_matrix, self.k, dim=1)
        sparse_weights = F.softmax(F.relu(S), dim=1)
        return torch.einsum('abc,ab->ac', self.mem_matrix[I], sparse_weights)

    def extra_repr(self):
        return 'mem_size={}, mem_dim={}'.format(self.mem_size, self.mem_dim)


class CascadeMemory(nn.Module):
    def __init__(self, mem_dim1, mem_dim2, param):
        super(CascadeMemory, self).__init__()
        self.mem_flag = 'CascadeMemory'
        self.memory = basicUNIT(param.MEM_SIZE, mem_dim1, param.ERASER_PROB, param.EPSILON)
        self.bn = nn.BatchNorm1d(mem_dim1)
        self.acti = nn.LeakyReLU()
        self.memory2 = basicUNIT(param.MEM_SIZE2, mem_dim2, param.ERASER_PROB2, param.EPSILON2)
        self.bn2 = nn.BatchNorm1d(mem_dim2)
        self.acti2 = nn.LeakyReLU()

    def forward(self, x):
        s = x.shape
        x = x.reshape(-1, s[-1])
        x = self.memory(x)
        x = self.bn(x)
        x = self.acti(x)
        x = self.memory2(x)
        x = self.bn2(x)
        x = self.acti2(x)
        x = x.reshape(s)
        return x


class CMmodel(nn.Module):
    def __init__(self, backbone, mem_dim=None, param=None):
        super().__init__()
        self.encoder = backbone.encoder
        self.decoder = backbone.decoder
        self.backbone_name = backbone.name
        self.latent_shape = backbone.latent_shape
        self.model_flag = type(self).__name__
        if mem_dim is None:
            self.mem_dim = math.prod(self.latent_shape)
        else:
            self.mem_dim = mem_dim

        self.fit = fit
        self.mem = CascadeMemory(self.mem_dim, self.mem_dim, param)

    def forward(self, x):
        orign = x
        x = x.reshape(-1, 784)
        x = self.encoder(x)
        x = self.mem(x.reshape(-1, self.mem_dim))
        x = x.reshape((-1,) + self.latent_shape)
        x = self.decoder(x)
        x = x.reshape(-1, 1, 28, 28)
        recon = x
        result = {'recon': x,
                  'total_loss': F.mse_loss(recon, orign),
                  }

        return result


def fit(model, train_loader, val_loader, test_loader, param):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
    writer = SummaryWriter('log/{}exp'.format(now))
    weight_path = '{0}/weights/{1}/run'.format(param.OUTPUT_PATH, model.model_flag)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    early_stopping = EarlyStopping(patience=param.PATIENCE, verbose=True,
                                   path='{}/checkpoint.pt'.format(weight_path), rho=param.RHO)
    optimizer = torch.optim.Adam(model.parameters(), lr=param.LR)
    train_losses_every_train_batch = []
    valid_losses_every_val_batch = []
    avg_train_losses = []
    avg_val_losses = []
    auc_losses = []
    train_start = time.time()
    train_batch_step_in_all = 0
    val_batch_step_in_all = 0
    for epoch in range(param.EPOCHS):
        model.train()
        model.mem.memory.module_state = 'training'
        model.mem.memory2.module_state = 'training'
        for data, _ in tqdm(train_loader, ncols=50, disable=param.DISABLE_TQDM):
            train_batch_step_in_all += 1
            data = data.float().to(param.DEVICE)
            output = model(data)
            loss = output['total_loss']
            writer.add_scalars("train_loss_items", {'recon': loss,
                                                    'lr': optimizer.param_groups[0]['lr'],
                                                    },
                               train_batch_step_in_all)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses_every_train_batch.append(loss.item())
        model.eval()
        model.mem.memory.module_state = 'valing'
        model.mem.memory2.module_state = 'valing'
        with torch.no_grad():
            for data, _ in tqdm(val_loader, ncols=50, disable=param.DISABLE_TQDM):
                val_batch_step_in_all += 1
                data = data.float().to(param.DEVICE)
                output = model(data)
                loss = output['total_loss']
                writer.add_scalar('val_loss', loss, val_batch_step_in_all)
                valid_losses_every_val_batch.append(loss.item())
        train_loss = np.average(train_losses_every_train_batch)
        valid_loss = np.average(valid_losses_every_val_batch)
        avg_train_losses.append(train_loss)
        avg_val_losses.append(valid_loss)
        epoch_len = len(str(epoch))
        auc = output_auc(model, test_loader, param)
        auc_losses.append(auc)
        print_msg = (f'[{epoch:>{epoch_len}}/{param.EPOCHS:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.9f} ' +
                     f'val_loss: {valid_loss:.9f}' +
                     f' AUC:{auc:.9f}')
        print(print_msg)
        train_losses_every_train_batch = []
        valid_losses_every_val_batch = []
        early_stopping(valid_loss, model)
        if early_stopping.stopFlag:
            print("[EarlyStop] Early stop.")
            break
    train_end = time.time()
    print("Run Time for training:", train_end - train_start)
    model.load_state_dict(torch.load('{}/checkpoint.pt'.format(weight_path)))
    writer.close()
    return model
