from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from functools import reduce
from typing import Tuple
import operator
import math

class MaskFunction(nn.Module):
    """
    Base class for mask functions.
    
    Attributes:
        hidden_dim (int): Hidden dimensionality of the MLP.
        features (int): Number of output mask channels.
        init_bias (float): Initial bias for the last layer.
    """
    def __init__(self, hidden_dim=32, features=3, init_bias=-0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.features = features
        self.init_bias = init_bias

    def forward(self, t, hard_sample=False):
        raise NotImplementedError("Subclasses should implement this method.")

# 3) Mask network: maps t→logits
class MaskFunctionMLP(MaskFunction):
    def __init__(self, hidden_dim=32, features=3, init_bias=-0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features)
        )

    def forward(self, t, hard_sample=False):
        # t: (B, T)

        if len(t.shape) == 2:
            B, T = t.shape
        else:
            T = t.shape[0]
            B = 1
            t = t.view(1, T)

        logits = self.mlp(t.contiguous().view(B*T, 1)).view(B, T, -1)

        if hard_sample:
            return (logits > 0.5).float()
        else:
            return logits

class MaskFunctionFourier(MaskFunction):
    def __init__(self, hidden_dim=64, features=3, init_bias=0, L=10):
        """
        features: number of output mask channels
        hidden_dim: hidden dimensionality of the MLP
        L: number of Fourier frequencies (will produce 2*L input dims)
        """
        super().__init__()
        self.L = L
        # build the MLP
        self.net = nn.Sequential(
            nn.Linear(2*L, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features)
        )

    def forward(self, t):
        """
        t: (B, T) tensor of scalars in [0,1]
        returns: (B, T, features)
        """
        if len(t.shape) == 2:
            B, T = t.shape
        else:
            T = t.shape[0]
            B = 1
            t = t.view(1, T)
        # build frequency vector [2π·2⁰, 2π·2¹, …, 2π·2⁽ᴸ⁻¹⁾]
        device = t.device
        freqs = 2*math.pi * (2.0 ** torch.arange(self.L, device=device))
        # expand t and multiply
        # t_lin: (B*T, L)
        t_lin = (t.view(-1,1) * freqs[None,:])
        # concat sin and cos → (B*T, 2L)
        fourier_feats = torch.cat([t_lin.sin(), t_lin.cos()], dim=-1)
        # feed through MLP
        out = self.net(fourier_feats)         # (B*T, features)
        return out.view(B, T, -1)

class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        # x: (..., 1)
        return torch.sin(self.w0 * x)

class MaskFunctionSine(MaskFunction):
    def __init__(self, hidden_dim=64, features=3, init_bias=0, w0=30.0):
        """
        features: number of output mask channels
        hidden: hidden dimensionality
        w0: frequency multiplier inside each sine
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            Sine(w0),
            nn.Linear(hidden_dim, hidden_dim),
            Sine(w0),
            nn.Linear(hidden_dim, features)
        )

    def forward(self, t):
        """
        t: (B, T) tensor of scalars in [0,1]
        returns: (B, T, features)
        """
        if len(t.shape) == 2:
            B, T = t.shape
        else:
            T = t.shape[0]
            B = 1
            t = t.view(1, T)
        x = t.view(B*T, 1)
        out = self.net(x)                     # (B*T, features)
        return out.view(B, T, -1)

class MaskFunctionHaar(MaskFunction):
    def __init__(self, features=3, hidden_dim=8, init_bias=0, levels=10):
        """
        features:    number of output mask channels
        hidden_dim:  hidden size of the MLP
        levels:      how many scales of Haar functions (total basis funcs = 2^levels - 1)
        """
        super().__init__()
        # total number of Haar basis functions (excluding the DC term)
        num_basis = sum(2**j for j in range(levels))

        # build lists of left/right half-interval boundaries
        starts_left  = []
        ends_left    = []
        starts_right = []
        ends_right   = []
        for j in range(levels):
            subdiv = 2**j
            width  = 1.0 / subdiv
            half   = width / 2.0
            for k in range(subdiv):
                a = k * width        # interval start
                b = a + half         # midpoint
                d = a + width        # interval end
                starts_left.append(a)
                ends_left.append(b)
                starts_right.append(b)
                ends_right.append(d)

        # register as fixed buffers (so they move to .cuda() with the model, etc.)
        self.register_buffer('starts_left',  torch.tensor(starts_left, dtype=torch.float32))
        self.register_buffer('ends_left',    torch.tensor(ends_left,   dtype=torch.float32))
        self.register_buffer('starts_right', torch.tensor(starts_right,dtype=torch.float32))
        self.register_buffer('ends_right',   torch.tensor(ends_right,  dtype=torch.float32))

        # simple MLP head
        self.net = nn.Sequential(
            nn.Linear(num_basis, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features),
        )

    def forward(self, t):
        """
        t: (B, T) tensor of time-points in [0,1]
        returns: (B, T, features)
        """
        if len(t.shape) == 2:
            B, T = t.shape
        else:
            T = t.shape[0]
            B = 1
            t = t.view(1, T)
        # flatten to shape (B*T,)
        t_flat = t.view(-1)

        # do one big broadcasted comparison:
        ml = (t_flat.unsqueeze(1) >= self.starts_left)  & (t_flat.unsqueeze(1) < self.ends_left)
        mr = (t_flat.unsqueeze(1) >= self.starts_right) & (t_flat.unsqueeze(1) < self.ends_right)

        # Haar feature = +1 on left half, -1 on right half, 0 elsewhere
        H = ml.float() - mr.float()   # shape (B*T, num_basis)

        # MLP & reshape
        out = self.net(H)             # (B*T, features)
        return out.view(B, T, -1)     # (B, T, features)

class MaskTensor(nn.Module):
    def __init__(self, data_tensor: torch.Tensor, init_value=0.5, learn=True):
        super().__init__()
        self.learn = learn
        if not learn:
            self.tensor = data_tensor
        else:
            self.tensor = nn.Parameter(torch.ones_like(data_tensor)*init_value)

    def forward(self, t, hard_sample=False):
        # hard_sample is mainly fr consistency with MaskFunction, yet to use it
        if t.shape[-1] != self.tensor.shape[-2]:
            raise Warning(f"Input tensor shape {t.shape} does not match mask tensor shape {self.tensor.shape}")
        if hard_sample:
            if not self.learn:
                return (self.tensor > 0.5).float()
            else:
                return (self.tensor.sigmoid() > 0.5).float()
        else:
            if not self.learn:
                return self.tensor
            else:
                return self.tensor.sigmoid()


def sample_hard_concrete(logits, temperature):
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y_soft = torch.sigmoid((logits + g) / temperature)
    y_hard = (y_soft > 0.5).int()
    return y_hard.detach() - y_soft.detach() + y_soft

class Perturbation:
    """Base class for perturbation in continuous time"""
    
    @abstractmethod
    def __init__(self, device, eps=1.0e-7):
        self.mask_function = None
        self.eps = eps
        self.device = device

    @abstractmethod
    def apply(self, t, X, data_mask, mask, K=1, temp=0.8):
        pass

    @abstractmethod
    def apply_multiple(self, t, X, data_mask, mask_function):
        pass

class Perturbation_continuous:
    """Base class for perturbation in continuous time"""
    
    @abstractmethod
    def __init__(self, device, eps=1.0e-7):
        self.mask_function = None
        self.eps = eps
        self.device = device

    @abstractmethod
    def apply(self, t, X, data_mask, mask, K=1, temp=0.8):
        pass

    @abstractmethod
    def apply_multiple(self, t, X, data_mask, mask_function):
        pass

class Deletion(Perturbation_continuous):
    """This class allows to create and apply deletion perturbations on inputs based on masks.
    
    Attributes:
        eps (float): Small number used for numerical stability.
        device: Device on which the tensor operations are executed.
    """
    def __init__(self, device, eps=1.0e-7):
        super().__init__(eps=eps, device=device)

    def adapt_data_mask(self, t, X, data_mask, deletion_mask):
        
        data_mask = data_mask * deletion_mask
        X[~deletion_mask.bool()] = torch.nan

        return t, X, data_mask
        
    def delete_and_pad_all(self,
                           t: torch.Tensor,
                           X: torch.Tensor,
                           tensor2: torch.Tensor,
                           deletion_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Remove time‑steps where deletion_mask is all zero (across features),
        then forward‑fill with the last kept row so that each output has the
        same shape as its input.

        Args:
        X, tensor2:      (..., T, F) tensors
        t:               (..., T) or (..., T, 1) tensor
        deletion_mask:   same shape as X (or tensor2)

        Returns:
        X_out, tensor2_out, t_out
        where
        - X_out, tensor2_out: (..., T, F)
        - t_out:              (..., T)  (same as original t’s shape)
        """
        # 1) unify t’s last dim
        if t.dim() == X.dim() - 1:
            t = t.unsqueeze(-1)   # now (..., T, 1)
        if t.shape[:-2] != X.shape[:-2] or t.shape[-2] != X.shape[-2]:
            raise ValueError("t must match X in all dims except its last, which may be 1")

        # 2) flatten leading batch dims → (N, T, Fi)
        *batch_shape, T, F = X.shape
        N = reduce(operator.mul, batch_shape, 1)

        tensor2[~deletion_mask.bool()] = 0

        flat = []
        for inp in (X, tensor2, t):
            Fi = inp.shape[-1]
            flat.append(inp.contiguous().view(N, T, Fi))
        Xf, Yf, tf = flat

        Mf = deletion_mask.view(N, T, deletion_mask.shape[-1])

        # 3) find which rows survive
        keep_any = Mf.any(dim=2)        # (N, T)
        lengths = keep_any.sum(dim=1)   # (N,)

        # 4) argsort the drop‐flag so kept rows bubble to front (stable → keeps their order)
        drop_flag  = (~keep_any).long()           # 0 for keep, 1 for drop
        sorted_idx = torch.argsort(drop_flag, dim=1, stable=True)  # (N, T)
        idx_exp    = sorted_idx.unsqueeze(2)      # (N, T, 1)

        outs = []
        for inp_f in (Xf, Yf, tf):
            Fi = inp_f.shape[2]
            idx_e = idx_exp.expand(-1, -1, Fi)       # (N, T, Fi)

            # 5) reorder so first `lengths[i]` rows are the surviving ones
            reordered = inp_f.gather(1, idx_e)       # (N, T, Fi)

            # 6) forward‑fill the tail with the last kept row
            last_idx     = (lengths - 1).clamp(min=0)        # (N,)
            last_idx_e   = last_idx[:, None].expand(-1, T)  # (N, T)
            last_idx_e3  = last_idx_e.unsqueeze(2).expand(-1, -1, Fi)
            last_vals    = reordered.gather(1, last_idx_e3) # (N, T, Fi)

            ar = torch.arange(T, device=X.device)[None, :]   # (1, T)
            pad_mask = ar >= lengths[:, None]               # (N, T)
            pad_mask = pad_mask.unsqueeze(2).expand(-1, -1, Fi)

            out_f = torch.where(pad_mask, last_vals, reordered)  # (N, T, Fi)
            outs.append(out_f)

        # 7) reshape back to original batch dims
        X_out, Y_out, t_out = [
            out.view(*batch_shape, T, out.shape[-1]) for out in outs
        ]

        # 8) restore t’s original shape
        if t_out.shape[-1] == 1 and t.dim() == X.dim() - 1:
            t_out = t_out.squeeze(-1)

        return t_out.squeeze(-1), X_out, Y_out
    
    def delete_and_fill_tail(
        t: torch.Tensor,
        X: torch.Tensor,
        data_mask: torch.Tensor,
        deletion_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Delete rows where all features are NaN, then append copies of the last valid row
        to restore the original length. In the appended rows, data_mask is set to zero.

        Vectorized implementation using argsort to move fully-missing rows to the end,
        followed by tail-filling from the last valid entry.

        Args:
            t (torch.Tensor): Tensor of shape (..., T)
            X (torch.Tensor): Tensor of shape (..., T, F)
            data_mask (torch.Tensor): Same shape as X, indicating observed data (1) or missing (0)
            deletion_mask (torch.Tensor): Same shape as X; 0 indicates to delete (set NaN), 1 keep

        Returns:
            tuple:
                - t_out: torch.Tensor of shape (..., T), with tail rows filled
                - X_out: torch.Tensor of shape (..., T, F), with tail rows filled
                - mask_out: torch.Tensor of shape (..., T, F), with tail rows zeroed
        """

        print('deleting and padding the tail')
        # Apply deletion mask
        X = X.masked_fill(deletion_mask == 0, float('nan'))
        data_mask = data_mask.masked_fill(deletion_mask == 0, 0)

        # Flatten batch dims
        lead = X.shape[:-2]
        T, F = X.shape[-2], X.shape[-1]
        B = int(torch.prod(torch.tensor(lead))) if lead else 1
        X_flat = X.reshape(B, T, F)
        t_flat = t.reshape(B, T)
        mask_flat = data_mask.reshape(B, T, F)

        # Identify fully-missing rows
        full_missing = torch.all(torch.isnan(X_flat), dim=-1).to(torch.int64)  # [B, T], 1 if missing

        # Stable sort requires PyTorch >=1.13; this groups valid(0) before missing(1)
        idx_sort = torch.argsort(full_missing, dim=1)  # [B, T]

        # Gather sorted sequences
        idx_sort_exp = idx_sort.unsqueeze(-1).expand(-1, -1, F)
        X_sorted = torch.gather(X_flat, 1, idx_sort_exp)    # [B, T, F]
        t_sorted = torch.gather(t_flat, 1, idx_sort)        # [B, T]
        mask_sorted = torch.gather(mask_flat, 1, idx_sort_exp)

        # Count valid entries per batch
        valid_counts = (full_missing == 0).sum(dim=1)      # [B]

        # Prepare tail fill
        # last valid per batch: at position valid_counts[b]-1
        last_idx = (valid_counts - 1).clamp(min=0)
        # gather last valid rows
        last_X = X_sorted[torch.arange(B), last_idx]       # [B, F]
        last_t = t_sorted[torch.arange(B), last_idx]       # [B]

        # Create mask for tail positions
        positions = torch.arange(T, device=X.device).unsqueeze(0).expand(B, -1)  # [B, T]
        tail_mask = positions >= valid_counts.unsqueeze(1)                     # [B, T]
        tail_mask_exp = tail_mask.unsqueeze(-1).expand(-1, -1, F)

        # Expand last valid to full shape
        last_X_exp = last_X.unsqueeze(1).expand(-1, T, -1)
        last_t_exp = last_t.unsqueeze(1).expand(-1, T)
        zeros_mask = torch.zeros_like(mask_sorted)

        # Apply tail fill
        X_filled_flat = torch.where(tail_mask_exp, last_X_exp, X_sorted)
        t_filled_flat = torch.where(tail_mask, last_t_exp, t_sorted)
        mask_filled_flat = torch.where(tail_mask_exp, zeros_mask, mask_sorted)

        # Reshape back
        X_out = X_filled_flat.reshape(*lead, T, F)
        t_out = t_filled_flat.reshape(*lead, T)
        mask_out = mask_filled_flat.reshape(*lead, T, F)

        return t_out, X_out, mask_out


    # TODO: Figure out what do to with K here
    def apply(self, t, X, data_mask, mask, K=1, temp=0.8, delete_and_pad=False):
        """
        Apply the deletion perturbation to a single time series.
        
        Parameters:
            t (torch.Tensor): 1D tensor of time stamps, shape (B, T,).
            X (torch.Tensor): Input data, shape (B, T, n_features).
            data_mask (torch.Tensor): Binary mask of shape (B, T, n_features) indicating observed data.
            mask_output (torch.Tensor)): same shape as X \in [0,1]
            
        Returns:
            torch.Tensor: Perturbed output, shape (T, n_features).
        """
        super().apply(t=t, X=X, data_mask=data_mask, mask=mask, K=K, temp=temp)
        
        mask_output = mask(t)

        if K > 1:
            mask_output = mask_output.unsqueeze(0).expand(K, -1, -1, -1)
            t = t.clone().unsqueeze(0).expand(K, -1, -1)
            X = X.clone().unsqueeze(0).expand(K, -1, -1, -1)
            data_mask = data_mask.clone().unsqueeze(0).expand(K, -1, -1, -1)

        z = sample_hard_concrete(mask_output, temperature=temp)

        if delete_and_pad:
            t_pert, X_pert, data_mask_pert = self.delete_and_fill_tail(t, X, data_mask, z)
        else:
            t_pert, X_pert, data_mask_pert = self.adapt_data_mask(t, X, data_mask, z)

        return t_pert, X_pert, data_mask_pert
    

class GaussianBlur(Perturbation):
    """
    This class applies Gaussian blur perturbations to inputs using two masks:
      - `tensor_mask` is used to compute the Gaussian width (sigma) for each time point and feature.
      - `data_mask` is a binary mask (same shape as X) that indicates which data points to consider.
    
    Attributes:
        eps (float): Small number for numerical stability.
        device: Device for tensor operations.
        sigma_max (float): Maximum width for the Gaussian blur.
    """
    def __init__(self, device, eps=1.0e-7, sigma_max=2):
        super().__init__(eps=eps, device=device)
        self.sigma_max = sigma_max

    def apply(self, t, X, data_mask, pert_mask, K=1, missing_value=0, temp=0.8):
        """
        Apply Gaussian blur to the input time series X.
        
        Parameters:
            t (torch.Tensor): 1D tensor of time points, shape (T,).
            X (torch.Tensor): Input data, shape (T, n_features).
            data_mask (torch.Tensor): Binary mask of shape (T, n_features); 0 indicates to ignore that data point.
            pert_mask (torch.Tensor or nn.Module):
            K is not used here, but is included for consistency with other methods.
            missing_value (float): Value to use for missing data points. Sometimes 0 is better, sometimes NaN.
                    We'll set this up so that the functions ignore the missing data but don't fail. Here,
                    torch.nan is convenient
        
        Returns:
            torch.Tensor: Blurred version of X, shape (T, n_features).
        """
        T = X.shape[-2]

        X[data_mask == 0] = missing_value  # Set ignored data points to the missing value.

        # Ensure t and data_mask are on the correct device.
        t = t.to(self.device)
        data_mask = data_mask.to(self.device)
        
        # Compute sigma per time point and feature using pert_mask.
        # A higher mask_tensor value will result in a smaller sigma (narrower Gaussian).
        if isinstance(pert_mask, MaskFunction):
            sigma_tensor = self.sigma_max * ((1 + self.eps) - Func.sigmoid(pert_mask(t=t))) # shape: (T, n_features)
        else:
            sigma_tensor = self.sigma_max * ((1 + self.eps) - pert_mask(t=t)) # shape: (T, n_features)

        sigma_tensor = sigma_tensor.unsqueeze(0)  # shape: (1, T, n_features) for broadcasting
        
        # Create tensors for time differences using the provided time points.
        t1 = t.view(T, 1, 1)  # shape: (T, 1, 1)
        t2 = t.view(1, T, 1)  # shape: (1, T, 1)
        time_diff_squared = (t1 - t2) ** 2  # shape: (T, T, 1)
        
        # Compute the Gaussian filter coefficients.
        # For each destination time point (index from t2) and for each source time point (index from t1),
        # the weight is based on the distance squared divided by the squared sigma.
        filter_coefs = torch.exp(- time_diff_squared / (2 * (sigma_tensor ** 2)))  # shape: (T, T, n_features)
        
        # Normalize the coefficients along the source time dimension (dimension 0).
        filter_coefs = filter_coefs / (torch.sum(filter_coefs, dim=1, keepdim=True) + self.eps)
        
        # Incorporate the data_mask: zero out contributions from data points to be ignored.
        # data_mask is expanded so that each source time point is multiplied by its corresponding mask value.
        filter_coefs = filter_coefs * data_mask.unsqueeze(1)  # shape remains (T, T, n_features)
        
        # Renormalize the coefficients after applying the data_mask.
        normalizer = torch.sum(filter_coefs, dim=1, keepdim=True) + self.eps
        filter_coefs = filter_coefs / normalizer
        
        # Compute the blurred output using Einstein summation:
        # For each destination time point, sum over source time points (weighted by the filter coefficients).
        X_pert = torch.einsum("bsti,bsi->bti", filter_coefs, X)
        return t, X_pert, data_mask
    
    

class FadeMovingAverage(Perturbation):
    """This class allows to create and apply 'fade to moving average' perturbations on inputs based on masks.
    
    Attributes:
        eps (float): Small number used for numerical stability.
        device: Device on which the tensor operations are executed.
    """
    def __init__(self, device, eps=1.0e-7):
        super().__init__(eps=eps, device=device)

    def apply(self, t, X, data_mask, pert_mask, K=1, temp=0.8):
        """
        Apply the fade-to-moving-average perturbation to a single time series.
        
        Parameters:
            t (torch.Tensor): 1D tensor of time stamps, shape (T,).
            X (torch.Tensor): Input data, shape (T, n_features).
            data_mask (torch.Tensor): Binary mask of shape (T, n_features) indicating observed data.
            pert_mask (torch.Tensor or nn.Module):
            K (int): Number of samples to generate (not used here, included for consistency).
            
        Returns:
            torch.Tensor: Perturbed output, shape (T, n_features).
        """
        # TODO: not clean here: should adapt the classes...
        super().apply(t=t, X=X, data_mask=data_mask, mask=pert_mask)
        T = X.shape[0]

        # Compute the moving average per feature using data_mask.
        # Only the observed points (where data_mask == 1) contribute.
        moving_avg = torch.sum(X * data_mask, dim=-2) / (torch.sum(data_mask, dim=-2) + self.eps)
        # Tile the moving average to the same time dimension as X.
        moving_average_tiled = moving_avg.view(1, -1).repeat(T, 1).to(self.device)
        # The perturbation is an affine blend of the original input and its moving average.
        
        if isinstance(pert_mask, MaskFunction):
            
            X_pert = Func.sigmoid(pert_mask(t=t)) * X + (1 - Func.sigmoid(pert_mask(t=t))) * moving_average_tiled
        else:
            X_pert = pert_mask(t=t) * X + (1 - pert_mask(t=t)) * moving_average_tiled
        
        return t, X_pert, data_mask