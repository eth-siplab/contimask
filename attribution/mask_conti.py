import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.nn.init as init
import math
import tqdm
from torch.optim.optimizer import Optimizer as TorchOptimizer

from evotorch.neuroevolution import NEProblem
from evotorch.algorithms import PGPE

from typing import Callable, Union

from attribution.perturbation_conti import Perturbation_continuous, Deletion, MaskFunction, MaskTensor

# TODO: test how choice of intergation influences convergence of model
def area_loss_func(pert_mask: Callable, time_lower_bound, time_upper_bound, res, device) -> torch.Tensor:

    ts = torch.linspace(time_lower_bound, time_upper_bound, math.ceil((time_upper_bound - time_lower_bound)/res), device=device)
    mask_vals = pert_mask(ts.unsqueeze(-1))

    return torch.trapz(mask_vals, dx=res, dim=0).mean()

# TODO: test how choice of intergation influences convergence, same as above
def absolute_derivate_loss_func(pert_mask: Callable, time_lower_bound, time_upper_bound, res, device) -> torch.Tensor:

    ts = torch.linspace(time_lower_bound, time_upper_bound, math.ceil((time_upper_bound - time_lower_bound)/res), device=device)
    mask_vals = pert_mask(ts.unsqueeze(-1))
    mask_vals_diffs = torch.diff(mask_vals, dim=0)
    
    return torch.trapz(torch.abs(mask_vals_diffs), dx=res, dim=0).mean()

def sample_hard_concrete(logits, temperature=0.1):
    """
    Gumbel-sigmoid (a.k.a. hard concrete) sampling with straight-through:
      - yields exact 0/1 samples in the forward pass
      - backprops through the *continuous* relaxation
    """
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y_soft = torch.sigmoid((logits + g) / temperature)
    y_hard = (y_soft > 0.5).float()
    return y_hard.detach() - y_soft.detach() + y_soft
       
class ContiMask(nn.Module):
    """
    Base class for continuous time masking.

    Args:
        forward_func (Callable): Function to compute the forward pass.
        model (nn.Module, optional): Model to be used. Defaults to None.
        batch_size (int, optional): Batch size for the model. Defaults to 32.
    """
    def __init__(self, forward_func: Callable,
                 perturbation_func: Callable = None,
                 model: nn.Module = None,
                 pert_mask: Union[nn.Module, torch.Tensor] = None,
                 device: str = "cpu",
                 ):
        
        super().__init__()
        self.forward_func = forward_func
        self.perturbation_func = perturbation_func
        self.device = device
        self.pert_mask = pert_mask
    
    def reset_parameters(self):
        return self.mask_function.reset_parameters()
    
    def get_mask(self, t):
        """
        Get the mask for the given time points.
        """
        if isinstance(self.pert_mask, MaskFunction) and isinstance(self.perturbation_func, Perturbation_continuous):
            print("returning for MaskFunction with continuous-time perturbation")
            return (self.pert_mask(t) > 0.5).float()
        elif isinstance(self.pert_mask, MaskFunction):
            print("returning for MaskFunction with discrete-time perturbation")
            return Func.sigmoid(self.pert_mask(t))
        elif isinstance(self.pert_mask, MaskTensor):
            print("returning for MaskTensor")
            return self.pert_mask(t)
        else:
            raise TypeError(f"pert_mask must be a nn.Module or a torch.Tensor, got {type(self.pert_mask)}")
    
    def attribute(self,
                  t: torch.Tensor, 
                  X: torch.Tensor, 
                  data_mask: torch.Tensor,
                  n_epoch: int,
                  lr: float = 0.01,
                  K=10,
                  lambda_l1: float = 0.01,
                  lambda_tv: float = 1,
                  plot_iter=False,
                  bernulli_temp_start: float = 0.9,
                  bernulli_temp_end: float = 0.7,
                  lambda_sharp: float = 0.01,
                  optimization_strategy: str = "adam",
                  rec_epsilon: float = 1e-10,
                  popsize=100,
                  radius_init=3,
                  center_learning_rate=0.5,
                  stdev_learning_rate=0.2,
                  target_area: float = None,
                  deletion_mode: str = False 
                  ) -> torch.Tensor:
        
        B, T, feat = X.shape

        self.t = t.clone().to(self.device)
        self.X = X.clone().to(self.device)
        self.data_mask = data_mask.clone().to(self.device)
        self.pert_mask = self.pert_mask.to(self.device)
        
        if deletion_mode:
            self.deletion_mode = -1
        else:
            self.deletion_mode = 1

        with torch.no_grad():
            self.y_orig = self.forward_func(t=t, X=X, data_mask=data_mask).to(self.device)
            if K > 1 and isinstance(self.perturbation_func, Perturbation_continuous):
                self.y_orig = self.y_orig.unsqueeze(0).expand(K, -1)
        
        if optimization_strategy == "gradient":
            # A few things depend on whether the mask is defined as a Network (nn.Module) or simply a Tensor of the same shape as X
            if isinstance(self.pert_mask, MaskFunction):
                optimizer = torch.optim.Adam(self.pert_mask.parameters(), lr=lr)
            elif isinstance(self.pert_mask, MaskTensor):
                optimizer = torch.optim.Adam(self.pert_mask.parameters(), lr=lr)
            else:
                raise ValueError("pert_mask must be a nn.Module or a torch.Tensor")
            
            if (issubclass(type(self.pert_mask), nn.Module) == False) and (K > 1):
                raise ValueError("IF K>1, pert_mask must be a nn.Module")

        self.hist = torch.zeros(4, 0)

        # TODO: introduce a new class for perturbation_func if it is a nn.Module. But only running with Deletion Perturbation for now.
        if issubclass(type(self.perturbation_func), nn.Module):
            optimizer.add_param_group({'params': self.perturbation_func.parameters()})

        def iteration(module):

            t_pert, X_pert, data_mask_pert = self.perturbation_func.apply(self.t, self.X, self.data_mask, module, K=K)
            y_pred = self.forward_func(t_pert, X_pert, data_mask_pert)

            rec_loss = Func.mse_loss(y_pred, self.y_orig, reduction="none")

            # check that rec_loss is not none
            assert rec_loss is not None

            if torch.isnan(rec_loss).any():
                print('rec_loss is nan')
                print('preds:')
                print(y_pred)
                print('original:')
                print(self.y_orig)
                print('t_pert:')
                print(t_pert)
                print('X_pert:')
                print(X_pert)
                print('data_mask_pert:')
                print(data_mask_pert)

            # check that rec_loss is not nan
            assert not torch.isnan(rec_loss).any()

            if isinstance(module, MaskFunction) and isinstance(self.perturbation_func, Perturbation_continuous):
                mask_at_t =(module(self.t) > 0.5).float()
                sparsity = mask_at_t.mean(dim=(-1, -2))
                if target_area is not None:
                    sparsity = torch.abs(sparsity - target_area)
                tv_loss = torch.mean(torch.abs(mask_at_t[..., 1:, :] - mask_at_t[..., :-1, :]), dim=(-1, -2))
                sharpness = - torch.abs(mask_at_t - 0.5).mean(dim=(-1, -2))
            elif isinstance(module, MaskFunction):
                mask_at_t = Func.sigmoid(module(self.t))
                sparsity = mask_at_t.mean(dim=(-1, -2))
                if target_area is not None:
                    sparsity = torch.abs(sparsity - target_area)
                tv_loss = torch.mean(torch.abs(mask_at_t[..., 1:, :] - mask_at_t[..., :-1, :]), dim=(-1, -2))
                sharpness = - torch.abs(mask_at_t - 0.5).mean(dim=(-1, -2))
            else:
                sparsity = module(self.t).mean(dim=(-1, -2))
                if target_area is not None:
                    sparsity = torch.abs(sparsity - target_area)
                tv_loss = torch.mean(torch.abs(module(self.t)[..., 1:, :] - module(self.t)[..., :-1, :]), dim=(-1, -2))
                sharpness = - torch.abs(module(self.t) - 0.5).mean(dim=(-1, -2))

            loss = (self.deletion_mode * rec_loss + 10*lambda_l1 * sparsity + lambda_tv * tv_loss).mean()

            self.hist = torch.cat((self.hist, torch.tensor([[self.deletion_mode * rec_loss.mean().detach().cpu()], [lambda_l1 * sparsity.mean().detach().cpu()], [lambda_tv * tv_loss.mean().detach().cpu()], [lambda_sharp * sharpness.mean().detach().cpu()]])), dim=1)

            return loss

        if optimization_strategy == "gradient":

            for i in tqdm.tqdm(range(n_epoch*2)):

                loss = iteration(module=self.pert_mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        elif optimization_strategy == "evotorch":
            
            problem = NEProblem("min", 
                        self.pert_mask,
                        iteration,
                        device=self.device)

            searcher = PGPE(
                problem,
                popsize=popsize,
                radius_init=radius_init,
                center_learning_rate=center_learning_rate,
                stdev_learning_rate=stdev_learning_rate,
                symmetric=True
                )
            
            # _ = StdOutLogger(searcher=searcher)

            # logger = PandasLogger(searcher)
            searcher.run(n_epoch*0.7)

            self.pert_mask = problem.parameterize_net(searcher.status["pop_best"])

            pass_on_params = searcher.status["pop_best"]

            def iteration(module):

                t_pert, X_pert, data_mask_pert = self.perturbation_func.apply(self.t, self.X, self.data_mask, module, K=K)
                y_pred = self.forward_func(t_pert, X_pert, data_mask_pert)

                rec_loss = Func.mse_loss(y_pred, self.y_orig, reduction="none")

                if isinstance(module, MaskFunction) and isinstance(self.perturbation_func, Perturbation_continuous):
                    mask_at_t =(module(self.t) > 0.5).float()
                    sparsity = mask_at_t.mean(dim=(-1, -2))
                    if target_area is not None:
                        sparsity = torch.abs(sparsity - target_area)
                    tv_loss = torch.mean(torch.abs(mask_at_t[..., 1:, :] - mask_at_t[..., :-1, :]), dim=(-1, -2))
                    sharpness = - torch.abs(mask_at_t - 0.5).mean(dim=(-1, -2))
                elif isinstance(module, MaskFunction):
                    mask_at_t = Func.sigmoid(module(self.t))
                    sparsity = mask_at_t.mean(dim=(-1, -2))
                    if target_area is not None:
                        sparsity = torch.abs(sparsity - target_area)
                    tv_loss = torch.mean(torch.abs(mask_at_t[..., 1:, :] - mask_at_t[..., :-1, :]), dim=(-1, -2))
                    sharpness = - torch.abs(mask_at_t - 0.5).mean(dim=(-1, -2))
                else:
                    sparsity = module(self.t).mean(dim=(-1, -2))
                    if target_area is not None:
                        sparsity = torch.abs(sparsity - target_area)
                    tv_loss = torch.mean(torch.abs(module(self.t)[..., 1:, :] - module(self.t)[..., :-1, :]), dim=(-1, -2))
                    sharpness = - torch.abs(module(self.t) - 0.5).mean(dim=(-1, -2))

                loss = (self.deletion_mode * rec_loss + 100* lambda_l1 * sparsity + lambda_tv * tv_loss).mean()

                self.hist = torch.cat((self.hist, torch.tensor([[self.deletion_mode * rec_loss.mean().detach().cpu()], [lambda_l1 * sparsity.mean().detach().cpu()], [lambda_tv * tv_loss.mean().detach().cpu()], [lambda_sharp * sharpness.mean().detach().cpu()]])), dim=1)

                return loss

            problem = NEProblem("min", 
                        self.pert_mask,
                        iteration,
                        device=self.device)

            searcher = PGPE(
                problem,
                popsize=popsize,
                radius_init=radius_init,
                center_learning_rate=center_learning_rate,
                stdev_learning_rate=stdev_learning_rate,
                symmetric=True,
                center_init=torch.stack([t for t in pass_on_params])
                )
            
            searcher.run(n_epoch*0.3)

            self.pert_mask = problem.parameterize_net(searcher.status["pop_best"])

        else:
            raise ValueError(f"Unknown optimization strategy: {optimization_strategy}")