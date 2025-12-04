import torch
import torch.nn.functional as Func
from torcheval.metrics.functional import binary_f1_score
import random
import os
from datetime import datetime

from attribution.perturbation_conti import Deletion, GaussianBlur, FadeMovingAverage, MaskFunctionMLP, MaskTensor, MaskFunctionFourier, MaskFunctionSine, MaskFunctionHaar
from attribution.mask_conti import ContiMask

import matplotlib.pyplot as plt

###################################################################################################################

# Explainer Naming Logic: Mask - Perturbation - Optimizer

# Masks:
# MaskFunctionFourier : MFF
# MaskFunctionSine: MFS
# MaskFunctionMLP: MFMLP
# MaskFunctionHaar: MFH
# MaskTensor: MT

# Perturbations:
# Deletion: D
# GaussianBlur: GB
# FadeMovingAverage: FMA

# Optimizers:
# evotorch: E
# gradient: G

explainers_options = ["MFH-FMA-G", "MFH-FMA-E", "MFH-GB-G", "MFH-GB-E", "MFH-D-E",
                      "MT-FMA-G", "MT-FMA-E", "MT-GB-G", "MT-GB-E",
                      "MFMLP-FMA-G", "MFMLP-FMA-E", "MFMLP-GB-G", "MFMLP-GB-E", "MFMLP-D-E",
                      "MFS-FMA-G", "MFS-FMA-E", "MFS-GB-G", "MFS-GB-E", "MFS-D-E",
                      "MFF-FMA-G", "MFF-FMA-E", "MFF-GB-G", "MFF-GB-E", "MFF-D-E",
                      "MFH-FMA-G", "MFH-FMA-E", "MFH-GB-G", "MFH-GB-E", "MFH-D-E"]

def run_experimet(seed: int = 0,
                  N_ex: int = 10,
                  explainers = ['MT-FMA-G'],
                  N_feat: int = 3,
                  N_time = 100,
                  N_select: int = 5,
                  device = 'cuda:2'):
    
    some_index = random.randint(0, 1000000)

    # PARAMS
    lambda_l1 = 0.1 / 10000
    lambda_tv = 0.1 / 100000
    lambda_sharp = 0.1 / 1000000
    n_epochs = 1000
    lr = 0.01
    hidden_dim = 128
    L = 12

    save_dict = {
        'some_index': some_index,
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'seed': seed,
        'N_ex': N_ex,
        'explainers': explainers,
        'N_feat': N_feat,
        'N_time': N_time,
        'N_select': N_select,
        'device': device,
        'lambda_l1': lambda_l1,
        'lambda_tv': lambda_tv,
        'lambda_sharp': lambda_sharp,
        'n_epochs': n_epochs,
        'lr': lr,
        'hidden_dim': hidden_dim,
    }

    # Create the directory if it doesn't exist
    os.makedirs(f"results/rare_time_temp/{some_index}/", exist_ok=True)

    with open(f"results/rare_time_temp/{some_index}/rare_time_temp_{some_index}_params.txt", "w") as f:
        for key, value in save_dict.items():
            f.write(f"{key}: {value}\n")
    
    for i in range(N_ex):

        torch.manual_seed(seed + i)
        X = torch.randn(1, N_time, N_feat) * 3
        t = torch.linspace(0, 1, N_time).unsqueeze(0)  # (B, T)
        data_mask = torch.ones_like(X)  # (B, T, F)

        X = X.to(device)
        t = t.to(device)
        data_mask = data_mask.to(device)

        dur_1 = 0.45
        dur_2 = 0.10
        dur_3 = 0.30

        start_1 = torch.rand(1, device=device).item() * (1 - dur_1)
        start_2 = torch.rand(1, device=device).item() * (1 - dur_2)
        start_3 = torch.rand(1, device=device).item() * (1 - dur_3)

        end_1 = start_1 + dur_1
        end_2 = start_2 + dur_2
        end_3 = start_3 + dur_3

        def f_to_explain(t, X, data_mask, return_mask=False):

            """
            t:          (B, T)
            X:          (B, T, F)   with F==3
            data_mask:  (B, T, F)
            
            Feature-specific intervals:
            - feature 0: keep t∈(0.2, 0.65)
            - feature 1: keep t∈(0.6, 0.7)
            - feature 2: keep t∈(0.1, 0.4)
            """
            
            # build per-feature interval masks of shape (B, T, F)
            # each unsqueeze(-1) makes (B, T, 1), then we concat along last dim
            m0 = ((t > start_1) & (t < end_1)).float().unsqueeze(-1)  # (B, T, 1)
            m1 = ((t > start_2) & (t < end_2)).float().unsqueeze(-1)
            m2 = ((t > start_3) & (t < end_3)).float().unsqueeze(-1)
            
            
            t = t.unsqueeze(-1).expand(X.size())
            mask = data_mask * torch.cat([m2, m1, m0], dim=-1).expand(X.size())
            # for j in range(N_feat):
            #     output[..., j] += ((t[..., j][mask[..., j]==1].diff(dim=-1))**2).mean()
            
            if return_mask:
                return torch.cat([m2, m1, m0], dim=-1).expand(X.size())
            
            return mask.sum(dim=(-1, -2))
            
        true_mask = f_to_explain(t, X, data_mask, return_mask=True)

        for explainer in explainers:
            print(f"running {explainer} for {i}th experiment")

            if explainer.split('-')[0] == 'MT':
                pert_mask = MaskTensor(data_tensor=X.to(device), init_value=0.5)
            elif explainer.split('-')[0] == 'MFMLP':
                pert_mask = MaskFunctionMLP(hidden_dim=hidden_dim, features=N_feat, init_bias=1).to(device)
            elif explainer.split('-')[0] == 'MFS':
                pert_mask = MaskFunctionSine(hidden_dim=hidden_dim, features=N_feat, init_bias=0).to(device)
            elif explainer.split('-')[0] == 'MFF':
                pert_mask = MaskFunctionFourier(hidden_dim=hidden_dim, features=N_feat, L=L, init_bias=1).to(device)
            elif explainer.split('-')[0] == 'MFH':
                pert_mask = MaskFunctionHaar(hidden_dim=hidden_dim, features=N_feat, init_bias=0, levels=12).to(device)
            else:
                raise ValueError(f"Unknown explainer: {explainer}")
            if explainer.split('-')[1] == 'D':
                pert = Deletion(device=device)
            elif explainer.split('-')[1] == 'GB':
                pert = GaussianBlur(device=device)
            elif explainer.split('-')[1] == 'FMA':
                pert = FadeMovingAverage(device=device)
            else:
                raise ValueError(f"Unknown perturbation: {explainer.split('-')[1]}")
            if explainer.split('-')[2] == 'G':
                optimization_strategy = 'gradient'
            elif explainer.split('-')[2] == 'E':
                optimization_strategy = 'evotorch'
            else:
                raise ValueError(f"Unknown optimization strategy: {explainer.split('-')[2]}")
            
            mask = ContiMask(forward_func=f_to_explain, perturbation_func=pert, pert_mask=pert_mask, device=device)
            mask.attribute(t=t.to(device), X=X.to(device), data_mask=data_mask.to(device),
                            n_epoch=n_epochs, lr=lr, plot_iter=False,
                            lambda_l1=lambda_l1,
                            lambda_tv=lambda_tv,
                            lambda_sharp=lambda_sharp,
                            optimization_strategy=optimization_strategy)
            
            mask_at_t = mask.get_mask(t.to(device))[0].float()
            sparsity = mask_at_t.mean(dim=(-1, -2))
            tv_loss = torch.mean(torch.abs(mask_at_t[..., 1:, :] - mask_at_t[..., :-1, :]), dim=(-1, -2))
            sharpness = - torch.abs(mask_at_t - 0.5).mean(dim=(-1, -2))

            torch.save(mask.get_mask(t.to(device)), f"results/rare_time_temp/{some_index}/rare_time_temp_{some_index}_{explainer}_cv{i}_fitted.pt")
            torch.save(mask.hist, f"results/rare_time_temp/{some_index}/rare_time_temp_{some_index}_{explainer}_cv{i}_hist.pt")
            torch.save(true_mask, f"results/rare_time_temp/{some_index}/rare_time_temp_{some_index}_{explainer}_cv{i}_true.pt")

def get_explainers_sub(explainers: list, mask: list=None, pert: list=None, optim: list=None):
    
    if mask is not None:
        explainers = [ex for ex in explainers if ex.split('-')[0] in mask]
    if pert is not None:
        explainers = [ex for ex in explainers if ex.split('-')[1] in pert]
    if optim is not None:
        explainers = [ex for ex in explainers if ex.split('-')[2] in optim]

    return explainers

if __name__ == "__main__":

    seed = 0
    N_ex = 5
    N_feat = 3
    N_time = 100
    N_select = 5
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    explainers_sub = get_explainers_sub(explainers_options, mask=['MFF'], pert=['GB', 'FMA', 'D'], optim=['E'])

    run_experimet(seed=seed,
                  N_ex=N_ex,
                  explainers=explainers_sub,
                  N_feat=N_feat,
                  N_time=N_time,
                  N_select=N_select,
                  device=device)
