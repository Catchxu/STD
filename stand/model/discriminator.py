import torch
import torch.nn as nn
# import torch.nn.utils.spectral_norm as SNorm1d
from .generator import GeneEncoder, ImageEncoder


# class LinearBlock(nn.Module):
#     def __init__(self, in_dim, out_dim, act: bool = True):
#         super().__init__()
#         self.linear = nn.Sequential(
#             SNorm1d(nn.Linear(in_dim, out_dim)),
#             nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity()
#         )

#     def forward(self, x):
#         return self.linear(x)


class Discriminator(nn.Module):
    def __init__(self, patch_size, in_dim, out_dim=[512, 256], z_dim=256):
        super().__init__()
        self.gene_dis = GeneEncoder(in_dim, out_dim)
        self.image_dis = ImageEncoder(patch_size, z_dim=z_dim)
        self.critic = nn.Linear(out_dim[-1]*z_dim, 1)

    def forward(self, g_block, feat_g, feat_p):
        dis_g = self.gene_dis(g_block, feat_g)
        dis_p = self.image_dis(g_block[1], feat_p)
        dis = self.critic(torch.cat([dis_g, dis_p], dim=1))
        return dis