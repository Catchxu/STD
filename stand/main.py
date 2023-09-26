import os
import dgl
import torch
import pandas as pd
import anndata as ad
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Dict
from torch.nn import functional as F

from .model import STNet
from .model import Discriminator
from ._utils import seed_everything, calculate_gradient_penalty


class DetectOutlier:
    def __init__(self, n_epochs: int = 10, batch_size: int = 128,
                 learning_rate: float = 2e-4, mem_dim: int = 2048,
                 shrink_thres: float = 0.01, temperature: float = 1,
                 n_critic: int = 1, GPU: bool = True,
                 patch_size: Optional[int] = None,
                 random_state: Optional[int] = None,
                 weight: Optional[Dict[str, float]] = None):
        if GPU:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                print("GPU isn't available, and use CPU to train ODBC-GAN.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.lr = learning_rate
        self.mem_dim = mem_dim
        self.shrink_thres = shrink_thres
        self.tem = temperature
        self.n_critic = n_critic

        if random_state is not None:
            seed_everything(random_state)

        if weight is None:
            self.weight = {'w_rec': 50, 'w_adv': 1, 'w_enc': 1, 'w_gp': 10}
        else:
            self.weight = weight
    
    def fit(self, ref_g: dgl.DGLGraph, weight_dir: Optional[str] = None, **kwargs):
        tqdm.write('Begin to fine-tune the model on normal spots...')

        self.g = ref_g
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        self.dataset = dgl.dataloading.DataLoader(
            self.g, self.g.nodes(), self.sampler, device=self.device,
            batch_size=self.batch_size, shuffle=True,
            drop_last=True, num_workers=4)
        
        in_dim = self.g.ndata['gene'].shape[1]
        self.D = Discriminator(self.patch_size, in_dim, **kwargs).to(self.device)
        self.G = STNet(self.patch_size, in_dim, thres=self.shrink_thres,
                       mem_dim=self.mem_dim, tem=self.tem).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.D_scaler = torch.cuda.amp.GradScaler()
        self.G_scaler = torch.cuda.amp.GradScaler()

        self.L1 = nn.L1Loss().to(self.device)
        self.L2 = nn.MSELoss().to(self.device)

        self.prepare(weight_dir)

        self.D.train()
        self.G.train()
        with self.dataset.enable_cpu_affinity(), tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description(f'Train Epochs')

                for _, _, blocks in self.dataset:
                    # Update discriminator for n_critic times
                    for _ in range(self.n_critic):
                        self.update_D(blocks)
                    
                    # Update generator for one time
                    self.update_G(blocks)

                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)

        tqdm.write('Fine-tuning has been finished.')
                    
    @torch.no_grad()
    def prepare(self, weight_dir: Optional[str]):
        if weight_dir:
            pre_weights = torch.load(weight_dir)
        else:
            pre_weights = torch.load(os.path.dirname(__file__) + '/model.pth')

        # Load the pre-trained weights for Encoder and Decoder
        self.G.load_state_dict({k: v for k, v in pre_weights.items()})

        # Initial the fake_g & fake_p in graph
        self.G.eval()
        dataset = dgl.dataloading.DataLoader(
            self.g, self.g.nodes(), self.sampler, device=self.device,
            batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=4)

        fake_gs, fake_ps = [], []
        for _, _, blocks in dataset:
            real_g = blocks[0].srcdata['gene']
            real_p = blocks[1].srcdata['patch']
            fake_g, fake_p = self.G.pretrain(blocks, real_g, real_p)
            fake_gs.append(fake_g.cpu().detach())
            fake_ps.append(fake_p.cpu().detach())

        self.g.ndata['fake_gene'] = torch.cat(fake_gs, dim=0)
        self.g.ndata['fake_patch'] = torch.cat(fake_ps, dim=0)

        # Initial the memory block with the normal embeddings
        sum_t = self.mem_dim/self.update_size
        t = 0
        while t < sum_t:
            for _, _, blocks in self.dataset:
                real_g = blocks[0].srcdata['gene']
                real_p = blocks[1].srcdata['patch']
                z, _, _ = self.G(blocks, real_g, real_p)
                self.G.Memory.update_mem(z)
                t += 1
    
    def update_D(self, blocks):
        self.opt_D.zero_grad()

        # get real data from blocks
        real_g = blocks[0].srcdata['gene']
        real_p = blocks[1].srcdata['patch']

        # get fake data from total graph
        fake_g = self.g.ndata['fake_gene'][blocks[0].srcnodes()].to(self.device)
        fake_p = self.g.ndata['fake_patch'][blocks[1].srcnodes()].to(self.device)

        d1 = self.D(blocks, real_g, real_p)
        self.D_scaler.scale(-torch.mean(d1)).backward()

        d2 = self.D(blocks, fake_g, fake_p)
        self.D_scaler.scale(torch.mean(d2)).backward()

        gp = calculate_gradient_penalty(blocks, real_g, real_p, fake_g, fake_p, self.D)
        self.D_scaler.scale(gp * self.weight['w_gp']).backward()

        # store discriminator loss for printing training information
        self.D_loss = -torch.mean(d1) + torch.mean(d2) + gp * self.weight['w_gp']

        self.D_scaler.step(self.opt_D)
        self.D_scaler.update()
    
    def update_G(self, blocks):
        self.opt_G.zero_grad()

        # get real data from blocks
        real_g = blocks[0].srcdata['gene']
        real_p = blocks[1].srcdata['patch']
        real_z, gen_g, gen_p = self.G(blocks, real_g, real_p)
        
        # update fake data in total graph
        self.g.ndata['fake_gene'][blocks[-1].dstnodes()] = gen_g.cpu().detach()
        self.g.ndata['fake_patch'][blocks[-1].dstnodes()] = gen_p.cpu().detach()

        # get updated fake data from total graph
        fake_g = self.g.ndata['fake_gene'][blocks[0].srcnodes()].to(self.device)
        fake_p = self.g.ndata['fake_patch'][blocks[1].srcnodes()].to(self.device)
        fake_z, _, _ = self.G(blocks, fake_g, fake_p)

        # discriminator provides feedback
        d = self.D(blocks, fake_g, fake_p)

        Loss_enc = self.L2(real_z ,fake_z)
        Loss_rec = (self.L1(blocks[-1].dstdata['gene'], gen_g) + 
                    self.L1(blocks[-1].dstdata['patch'], gen_p))/2
        Loss_adv = -torch.mean(d)

        self.G_loss = (self.weight['w_enc']*Loss_enc +
                       self.weight['w_rec']*Loss_rec +
                       self.weight['w_adv']*Loss_adv)

        self.G_scaler.scale(self.G_loss).backward(retain_graph=True)
        self.G_scaler.step(self.opt_G)
        self.G_scaler.update()

        # updating memory block with generated embeddings, fake_z
        self.G.Memory.update_mem(fake_z)





        



        