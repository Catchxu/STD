import dgl
import torch
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from math import e
from PIL import Image
from typing import Literal, Optional, List
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors

from ._utils import seed_everything


class Build_graph:
    def __init__(self, adata: ad.AnnData, image: np.ndarray,
                 position: np.ndarray, n_neighbors: int = 4,
                 patch_size: int = 48, train_mode: bool = True):
        self.adata = adata
        self.image = image
        self.position = position
        self.n_neighbors = n_neighbors
        self.patch_size = patch_size
        self.train_mode = train_mode

        u, v = self.get_edge()
        self.g = dgl.to_bidirected(dgl.graph((u, v)))
        self.g = dgl.add_self_loop(self.g)

        self.g.ndata['patch'] = self.get_patch()
        self.g.ndata['gene'] = self.get_gene()

    def get_edge(self):
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1)
        nbrs = nbrs.fit(self.position)
        _, indices = nbrs.kneighbors(self.position)
        u = indices[:, 0].repeat(self.n_neighbors)
        v = indices[:, 1:].flatten()
        return u, v

    def get_patch(self):
        if not isinstance(self.image[0, 0, 0], np.uint8):
            self.image = np.uint8(self.image * 255)
            
        img = Image.fromarray(self.image)
        r = np.ceil(self.patch_size/2).astype(int)

        trans = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180)
        ])
 
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        p_list = []
        for i in range(len(self.position)):
            x, y = self.position[i, :]
            p = img.crop((x - r, y - r, x + r, y + r))
            if self.train_mode:
                p = trans(p)
            p = preprocess(p)
            p_list.append(p.reshape(3, 2*r, 2*r))
        return torch.stack(p_list)

    def get_gene(self):
        X = torch.Tensor(self.adata.X)
        X_max, X_min = torch.max(X, 1)[0], torch.min(X, 1)[0]
        dst = (X_max - X_min).unsqueeze(1)
        X_min = X_min.unsqueeze(1)
        X_norm = torch.sub(X, X_min).true_divide(dst)
        X_norm = (X_norm - 0.5).true_divide(0.5)
        return X_norm


class Build_multi_graph:
    def __init__(self, adata: List[ad.AnnData], image: List[np.ndarray],
                 position: List[np.ndarray], n_neighbors: int = 4,
                 patch_size: int = 48, train_mode: bool = True):
        self.adata = adata
        self.adata_raw = adata
        self.image = image
        self.position = position
        self.n_dataset = len(adata)
        self.n_neighbors = n_neighbors
        self.patch_size = patch_size
        self.train_mode = train_mode

        self.batch = self.get_batch()
        u, v = self.get_edge()
        self.g = dgl.to_bidirected(dgl.graph((u, v)))
        self.g = dgl.add_self_loop(self.g)

        self.g.ndata['batch'] = self.batch
        self.g.ndata['patch'] = self.get_patch()
        self.g.ndata['gene'] = self.get_gene()

    def get_batch(self):
        adata = []
        for i in range(self.n_dataset):
            a = self.adata[i]
            a.obs['batch'] = i
            adata.append(a)
        self.adata = ad.concat(adata, merge='same')
        self.adata.obs_names_make_unique(join=',')
        batch = np.array(pd.get_dummies(self.adata.obs['batch']), dtype=np.float32)
        return torch.Tensor(batch)

    def get_edge(self):
        self.adata.obs['idx'] = range(self.adata.n_obs)
        u_list, v_list = [], []
        for i in range(self.n_dataset):
            adata = self.adata[self.adata.obs['batch'] == i]
            position = self.position[i]
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1)
            nbrs = nbrs.fit(position)
            _, indices = nbrs.kneighbors(position)
            u = adata.obs['idx'][indices[:, 0].repeat(self.n_neighbors)]
            v = adata.obs['idx'][indices[:, 1:].flatten()]
            u_list = u_list + u.tolist()
            v_list = v_list + v.tolist()
        return u_list, v_list

    def get_patch(self):
        p_list = []
        for i in range(self.n_dataset):
            img = self.image[i]
            if not isinstance(img[0, 0, 0], np.uint8):
                img = np.uint8(img * 255)
            img = Image.fromarray(img)

            position = self.position[i]
            r = np.ceil(self.patch_size/2).astype(int)

            trans = transforms.Compose([
                transforms.ColorJitter(0.5, 0.5, 0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=180)
            ])
            
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            for i in range(len(position)):
                x, y = position[i, :]
                p = img.crop((x - r, y - r, x + r, y + r))
                if self.train_mode:
                    p = trans(p)
                p = preprocess(p)
                p_list.append(p.reshape(3, 2*r, 2*r))
        return torch.stack(p_list)
    
    def get_gene(self):
        X = torch.Tensor(self.adata.X)
        X_max, X_min = torch.max(X, 1)[0], torch.min(X, 1)[0]
        dst = (X_max - X_min).unsqueeze(1)
        X_min = X_min.unsqueeze(1)
        X_norm = torch.sub(X, X_min).true_divide(dst)
        X_norm = (X_norm - 0.5).true_divide(0.5)
        return X_norm


def preprocess_data(adata: ad.AnnData):
    adata = adata[:, adata.var_names.notnull()]
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata, base=e)
    return adata


def set_patch(adata: ad.AnnData):
    info = next(iter(adata.uns['spatial'].values()))['scalefactors']
    patch_size = info['fiducial_diameter_fullres']*info['tissue_hires_scalef']
    patch_size = int(patch_size)
    n = np.ceil(np.log(patch_size)/np.log(2)).astype(int)
    patch_size = 2**n
    return patch_size


def read(data_dir: str, data_name: str, preprocess: bool = True,
         return_type: Literal['anndata', 'graph'] = 'graph',
         n_neighbors: int = 4, patch_size: Optional[int] = None,
         train_mode: bool = True):
    seed_everything(0)
    input_dir = data_dir + data_name + '.h5ad'
    adata = sc.read(input_dir)
    image = next(iter(adata.uns['spatial'].values()))['images']['hires']
    position = adata.obsm['position']

    if preprocess:
        adata = preprocess_data(adata)

    if return_type == 'anndata':
        return adata, image, position
    elif return_type == 'graph':
        if patch_size is None:
            patch_size = set_patch(adata)
        graph = Build_graph(adata, image, position, n_neighbors,
                            patch_size, train_mode).g
        return graph


def read_cross(ref_dir: str, tgt_dir:str, ref_name: str, tgt_name: str, 
               preprocess: bool = True, n_genes: int = 3000, patch_size: Optional[int] = None,
               return_type: Literal['anndata', 'graph'] = 'graph', **kwargs):
    seed_everything(0)
    ref, ref_img, ref_pos = read(ref_dir, ref_name, preprocess=False, return_type='anndata')
    tgt, tgt_img, tgt_pos = read(tgt_dir, tgt_name, preprocess=False, return_type='anndata')
    overlap_gene = list(set(ref.var_names) & set(tgt.var_names))
    ref = ref[:, overlap_gene]
    tgt = tgt[:, overlap_gene]

    if preprocess:
        ref = preprocess_data(ref)
        tgt = preprocess_data(tgt)
        if len(overlap_gene) <= n_genes:
            warnings.warn(
                'There are too few overlapping genes to perform feature selection'
            )
        else:
            sc.pp.highly_variable_genes(ref, n_top_genes=n_genes, subset=True)
            tgt = tgt[:, ref.var_names]
    
    if return_type == 'anndata':
        return (ref, ref_img, ref_pos), (tgt, tgt_img, tgt_pos)

    elif return_type == 'graph':
        if patch_size is None:
            patch_size = set_patch(ref)
        ref_g = Build_graph(ref, ref_img, ref_pos,
                            patch_size=patch_size, **kwargs).g
        tgt_g = Build_graph(tgt, tgt_img, tgt_pos,
                            patch_size=patch_size, **kwargs).g
        return ref_g, tgt_g


def read_multi_graph(input_dir: str, data_name: List[str], patch_size: Optional[int] = None,
                     preprocess: bool = True, n_genes: int = 3000, **kwargs):
    seed_everything(0)
    adatas, images, positions = [], [], []
    for d in data_name:
        adata, image, position = read(input_dir, d, preprocess=False, return_type='anndata')
        adatas.append(adata)
        images.append(image)
        positions.append(position)
    
    if preprocess:
        adatas = [preprocess_data(d) for d in adatas]
        ref = adatas[0]
        sc.pp.highly_variable_genes(ref, n_top_genes=n_genes, subset=True)
        adatas = [d[:, ref.var_names] for d in adatas]
    
    if patch_size is None:
        patch_size = set_patch(adata)
    g = Build_multi_graph(adatas, images, positions,
                          patch_size=patch_size, **kwargs).g
    return g


