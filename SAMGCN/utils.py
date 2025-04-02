import sklearn
from natsort import natsorted
from typing import Optional
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse import issparse, csr_matrix
import torch.nn.functional as F
import sys
import rpy2.robjects as robjects
from contextlib import redirect_stdout, redirect_stderr
import io
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from anndata import AnnData
import torch as th
import scipy.sparse as sp
import sklearn
import torch
import networkx as nx
from sklearn.cluster import KMeans
import community as community_louvain
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import normalized_mutual_info_score
def spatial_reconstruction(
        adata_ori: AnnData,
        alpha: float = 1,
        n_neighbors: int = 10,
        n_pcs: int = 15,
        use_highly_variable: Optional[bool] = True,
        normalize_total: bool = True,  # 使用总数归一化
        copy: bool = True,
        n_components: int = 20,
):
    adata = adata_ori.copy() if copy else adata_ori
    adata.layers['counts'] = adata.X  # adata.X是表达矩阵

    sc.pp.normalize_total(adata) if normalize_total else None
    sc.pp.log1p(adata)  # log(1+x)对偏度比较大的数据用log1p函数进行转化，使其更加服从高斯分布。
    adata.layers['log1p-ori'] = adata.X

    hvg = list(adata.var['highly_variable'][adata.var['highly_variable'].values].index)
    exmatrix_ori = adata.to_df(layer='log1p-ori')[hvg].to_numpy()
    pca_ori = PCA(n_components=n_components)
    pca_ori.fit(exmatrix_ori)
    exmatrix_ori = pca_ori.transform(exmatrix_ori)

    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_highly_variable)

    coord = adata.obsm['spatial']  # 4727*2 (x,y)

    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(coord)
    nbrs = neigh.kneighbors_graph(coord)  # 4727*4727 邻接矩阵？

    dists = np.exp(2 - cosine_distances(adata.obsm['X_pca'])) - 1

    conns = nbrs.T.toarray() * dists

    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X_rec = alpha * np.matmul(conns / np.sum(conns, axis=0, keepdims=True), X) + X

    stg = conns / np.sum(conns, axis=0, keepdims=True)

    adata.X = csr_matrix(X_rec)
    adata.layers['log1p-aug'] = adata.X

    exmatrix_aug = adata.to_df(layer='log1p-aug')[hvg].to_numpy()
    pca_aug = PCA(n_components=n_components)
    pca_aug.fit(exmatrix_aug)
    exmatrix_aug = pca_ori.transform(exmatrix_aug)

    del adata.obsm['X_pca']

    '''记录重构参数'''
    adata.uns['spatial_reconstruction'] = {}
    rec_dict = adata.uns['spatial_reconstruction']
    rec_dict['params'] = {}
    rec_dict['params']['alpha'] = alpha
    rec_dict['params']['n_neighbors'] = n_neighbors
    rec_dict['params']['n_pcs'] = n_pcs
    rec_dict['params']['use_highly_variable'] = use_highly_variable
    rec_dict['params']['normalize_total'] = normalize_total

    return adata if copy else None, exmatrix_ori, exmatrix_aug, stg

def spatial_construct_graph1(adata, radius=950):

    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A=np.zeros((coor.shape[0],coor.shape[0]))

    # print("coor:", coor)
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]]=1

    print('The graph contains %d edges, %d cells.' % (sum(sum(A)), adata.n_obs))
    print('%.4f neighbors per cell on average.' % (sum(sum(A)) / adata.n_obs))

    graph_nei = torch.from_numpy(A)

    graph_neg = torch.ones(coor.shape[0],coor.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    # nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    return sadj, graph_nei, graph_neg#, nsadj
def spatial_construct_graph(positions, k=15):
    print("start spatial construct graph")
    A = euclidean_distances(positions)
    tmp = 0
    mink = 2
    for t in range(100, 1000, 100):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    for t in range(tmp - 100, 1000, 10):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    for t in range(tmp - 10, 1000, 5):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            A = A1
            break
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    # index = np.argwhere(A > 0)
    # np.savetxt('./result/edge.csv', index, delimiter=',')

    graph_nei = torch.from_numpy(A)
    # print(type(graph_nei),graph_nei)
    graph_neg = torch.ones(positions.shape[0], positions.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    # nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    return sadj, graph_nei, graph_neg#, nsadj

def features_construct_graph(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    print("start features construct graph")
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)
    # print("k: ", k)
    # print("features_construct_graph features", features.shape)
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    # index = np.argwhere(A > 0)
    # np.savetxt('./result/fadj.csv', index, delimiter=',')
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    # nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    # nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return fadj#, nfadj

def dopca(data, dim=50):
    return PCA(n_components=dim).fit_transform(data)


def normalize_sparse_matrix(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)






def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    #     Works with pytorch <= 1.2
    #     p_i_j[(p_i_j < EPS).data] = EPS
    #     p_j[(p_j < EPS).data] = EPS
    #     p_i[(p_i < EPS).data] = EPS

    # Works with pytorch > 1.2
    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss*-1

def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j

def refine_spatial_domains(y_pred, coord, n_neighbors=6):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coord)
    distances, indices = nbrs.kneighbors(coord)
    indices = indices[:, 1:]

    y_refined = pd.Series(index=y_pred.index, dtype='object')

    for i in range(y_pred.shape[0]):

        y_pred_count = y_pred[indices[i, :]].value_counts()

        if (y_pred_count.loc[y_pred[i]] < n_neighbors / 2) and (y_pred_count.max() > n_neighbors / 2):
            y_refined[i] = y_pred_count.idxmax()
        else:
            y_refined[i] = y_pred[i]

    y_refined = pd.Categorical(
        values=y_refined.astype('U'),
        categories=natsorted(map(str, y_refined.unique())),
    )
    return y_refined


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='SAmgcn', random_seed=100):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    # 捕获 stdout 和 stderr 输出
    f_stdout = io.StringIO()
    f_stderr = io.StringIO()
    with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)

    # 打印 res 的内容以检查其组件
    # print("Mclust result components:", res.names)

    # 检查 'classification' 组件是否存在
    if 'classification' in res.names:
        mclust_res = np.array(res.rx2('classification'))
    else:
        raise ValueError("Mclust result does not contain 'classification' component")

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def purity_func(cluster, label):
    cluster = np.array(cluster)
    label = np.array(label)
    indedata1 = {}
    for p in np.unique(label):
        indedata1[p] = np.argwhere(label == p)
    indedata2 = {}
    for q in np.unique(cluster):
        indedata2[q] = np.argwhere(cluster == q)

    count_all = []
    for i in indedata1.values():
        count = []
        for j in indedata2.values():
            a = np.intersect1d(i, j).shape[0]
            count.append(a)
        count_all.append(count)
    return sum(np.max(count_all, axis=0)) / len(cluster)


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)
def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

# def regularization_loss(emb, graph_nei, graph_neg):
#     mat = torch.sigmoid(cosine_similarity(emb))  # .cpu()
#     # mat = pd.DataFrame(mat.cpu().detach().numpy()).values
#
#     # graph_neg = torch.ones(graph_nei.shape) - graph_nei
#
#     neigh_loss = torch.mul(graph_nei, torch.log(mat)).mean()
#     neg_loss = torch.mul(graph_neg, torch.log(1 - mat)).mean()
#     pair_loss = -(neigh_loss + neg_loss) / 2
#     return pair_loss
def regularization_loss(emb, adj):
    mat = torch.sigmoid(cosine_similarity(emb))  # .cpu()
    loss = torch.mean((mat - adj) ** 2)
    return loss


def consistency_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    return torch.mean((cov1 - cov2) ** 2)


def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat
def loss_kld(mu, logvar):
    """
    KL divergence of normal distribution N(mu, exp(logvar)) and N(0, 1)

    Parameters
    ------
    mu
        mean vector of normal distribution
    logvar
        Logarithmic variance vector of normal distribution

    Returns
    ------
    KLD
        KL divergence loss
    """

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5) / mu.shape[0]

    return KLD


class NB(object):
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result

def contrastive_loss(z, positive_emb, mask_nega, n_nega, device, temperature=0.5):
        device = device
        # embedding L2 normalization
        emb = F.normalize(z, dim=-1, p=2)
        similarity = torch.matmul(emb, torch.transpose(emb, 1, 0).detach())  # cosine similarity matrix
        e_sim = torch.exp((similarity / temperature))

        positive_emb_norm = F.normalize(positive_emb, dim=-1, p=2).to(device)
        positive_sim = torch.exp((positive_emb_norm * emb.unsqueeze(1)).sum(axis=-1) / temperature)

        x = mask_nega._indices()[0]
        y = mask_nega._indices()[1]
        N_each_spot = e_sim[x, y].reshape((-1, n_nega)).sum(dim=-1)

        N_each_spot = N_each_spot.unsqueeze(-1).repeat([1, positive_sim.shape[1]])

        contras = -torch.log(positive_sim / (positive_sim + N_each_spot))

        return torch.mean(contras)




