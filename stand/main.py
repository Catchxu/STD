import os
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Dict
from torch.nn import functional as F

from .model import STNet
from .model import Discriminator
from ._utils import seed_everything, calculate_gradient_penalty


class STand:
    def __init__(self, n_epochs: int = 10, batch_size: int = 128,
                 learning_rate: float = 2e-5, mem_dim: int = 1024,
                 shrink_thres: float = 0.01, temperature: float = 1,
                 n_critic: int = 2, GPU: bool = True,
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
        self.lr = learning_rate
        self.mem_dim = mem_dim
        self.shrink_thres = shrink_thres
        self.tem = temperature
        self.n_critic = n_critic

        if random_state is not None:
            seed_everything(random_state)

        if weight is None:
            self.weight = {'w_rec': 30, 'w_adv': 1, 'w_gp': 10}
        else:
            self.weight = weight

    def fit(self, ref_g: dgl.DGLGraph, weight_dir: Optional[str] = None, **kwargs):
        '''Fine-tune STand on reference graph'''
        tqdm.write('Begin to fine-tune the model on normal spots...')

        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        self.dataset = dgl.dataloading.DataLoader(
            ref_g, ref_g.nodes(), self.sampler,
            batch_size=self.batch_size, shuffle=True,
            drop_last=True, num_workers=4, device=self.device)  # dataset provides subgraph for training

        self.in_dim = ref_g.ndata['gene'].shape[1]
        self.patch_size = ref_g.ndata['patch'].shape[2]
        self.n_batch = int(len(ref_g.nodes())/self.batch_size)
        self.D = Discriminator(self.patch_size, self.in_dim).to(self.device)
        self.G = STNet(self.patch_size, self.in_dim, thres=self.shrink_thres,
                       mem_dim=self.mem_dim, tem=self.tem, **kwargs).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr*2, betas=(0.5, 0.999))
        self.D_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer = self.opt_D,
                                                          T_max = self.n_batch*self.n_epochs*self.n_critic)
        self.G_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer = self.opt_G,
                                                          T_max = self.n_batch*self.n_epochs)
        self.D_scaler = torch.cuda.amp.GradScaler()
        self.G_scaler = torch.cuda.amp.GradScaler()

        self.L1 = nn.L1Loss().to(self.device)

        self.prepare(weight_dir)

        self.D.train()
        self.G.train()
        with tqdm(total=self.n_epochs) as t:
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
    def predict(self, tgt_g: dgl.DGLGraph):
        '''Detect anomalous spots on target graph'''
        if (self.G is None or self.D is None):
            raise RuntimeError('Please fine-tune the model first.')

        dataset = dgl.dataloading.DataLoader(
            tgt_g, tgt_g.nodes(), self.sampler,
            batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=4, device=self.device)

        self.G.eval()
        self.D.eval()
        tqdm.write('Detect anomalous spots on test dataset...')
    
        # calucate anomaly score
        dis = []
        for _, _, blocks in dataset:
            # get real data from blocks
            real_g = blocks[0].srcdata['gene']
            real_p = blocks[1].srcdata['patch']

            _, fake_g, fake_p = self.G(blocks, real_g, real_p)
            d = self.D(fake_g, fake_p)
            dis.append(d.cpu().detach())

        # Normalize anomaly scores
        dis = torch.mean(torch.cat(dis, dim=0), dim=1).numpy()
        score = (dis.max() - dis)/(dis.max() - dis.min())

        tqdm.write('Anomalous spots have been detected.\n')
        return list(score.reshape(-1))

    @torch.no_grad()
    def prepare(self, weight_dir: Optional[str]):
        '''Prepare stage for pretrained weights and memory block'''
        if weight_dir:
            pre_weights = torch.load(weight_dir)
        else:
            pre_weights = torch.load(os.path.dirname(__file__) + '/model.pth')

        # Load the pre-trained weights for Encoder and Decoder
        model_dict = self.G.state_dict()
        pretrained_dict = {k: v for k, v in pre_weights.items() if 'Memory' not in k}
        model_dict.update(pretrained_dict)
        self.G.load_state_dict(model_dict)

        # Initial the memory block with the normal embeddings
        sum_t = self.mem_dim/self.batch_size
        t = 0
        while t < sum_t:
            for _, _, blocks in self.dataset:
                real_g = blocks[0].srcdata['gene']
                real_p = blocks[1].srcdata['patch']
                z, _, _ = self.G(blocks, real_g, real_p)
                self.G.Memory.update_mem(z)
                t += 1
    
    def update_D(self, blocks):
        '''Updating discriminator'''
        self.opt_D.zero_grad()

        # generate fake data
        _, fake_g, fake_p = self.G(blocks,
                                   blocks[0].srcdata['gene'],
                                   blocks[1].srcdata['patch'])

        # get real data from blocks
        real_g = blocks[1].dstdata['gene']
        real_p = blocks[1].dstdata['patch']

        d1 = torch.mean(self.D(real_g, real_p))
        d2 = torch.mean(self.D(fake_g.detach(), fake_p.detach()))
        gp = calculate_gradient_penalty(real_g, real_p, fake_g.detach(), fake_p.detach(), self.D)

        # store discriminator loss for printing training information
        self.D_loss = - d1 + d2 + gp * self.weight['w_gp']
        self.D_scaler.scale(self.D_loss).backward()

        self.D_scaler.step(self.opt_D)
        self.D_scaler.update()
        self.D_sch.step()

    def update_G(self, blocks):
        '''Updating generator'''
        self.opt_G.zero_grad()

        # get real data from blocks
        real_g = blocks[0].srcdata['gene']
        real_p = blocks[1].srcdata['patch']
        real_z, fake_g, fake_p = self.G(blocks, real_g, real_p)
        
        # discriminator provides feedback
        d = self.D(fake_g, fake_p)

        Loss_rec = (self.L1(blocks[-1].dstdata['gene'], fake_g) + 
                    self.L1(blocks[-1].dstdata['patch'], fake_p))/2
        Loss_adv = -torch.mean(d)

        # store discriminator loss for printing training information and backward
        self.G_loss = (self.weight['w_rec']*Loss_rec +
                       self.weight['w_adv']*Loss_adv)

        self.G_scaler.scale(self.G_loss).backward()
        self.G_scaler.step(self.opt_G)
        self.G_scaler.update()
        self.G_sch.step()

        # updating memory block with generated embeddings, fake_z
        self.G.Memory.update_mem(real_z)






        



        