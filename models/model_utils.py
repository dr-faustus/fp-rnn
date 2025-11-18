import torch
import torch.nn as nn
from typing import Dict


def shift(input:torch.Tensor, shift, dim=0, fillval=0):
    # torch.roll without the copy of the wrap-around section
    size = input.size(dim)
    fill = torch.full_like(input.narrow(dim, 0, abs(shift)), fillval)
    if shift > 0:
        output = torch.cat([fill, input.narrow(dim, 0, size-shift)], dim=dim)
    if shift < 0:
        output = torch.cat([input.narrow(dim, -shift, size+shift), fill], dim=dim)
    return output

class FixedPointOptimizer(nn.Module):
    def __init__(self, stepsize=1, stepsize_decay=0.9, decay_patience=5, thresh=0.1, 
                 outlier_quantile=0.25, stepsize_thresh=1e-3, max_iter=100, eps=1e-8):
        super().__init__()
        self.stepsize = stepsize
        self.stepsize_decay = stepsize_decay
        self.decay_patience = decay_patience
        self.thresh = thresh
        self.outlier_quantile=outlier_quantile
        self.stepsize_thresh = stepsize_thresh
        self.max_iter = max_iter
        self.eps = eps
    
    def start(self, y:torch.Tensor):
        '''optimizer start: initialize state'''
        B, device = y.shape[0], y.device
       
        state = dict(y=y, #converged = torch.zeros(B).bool().to(device),
                     residues = torch.inf * torch.ones(B).to(device),
                     stepsize = self.stepsize * torch.ones(B, 1, 1).to(device),
                     patience = self.decay_patience * torch.ones(B).to(device),
                     iter_idx = torch.tensor(0).to(device))
        
        return state

    def step(self, state:Dict[str, torch.Tensor], y:torch.Tensor):
        '''optimizer step: update state'''
        residues = (state['y'] - y).norm(p=torch.inf, dim=-1) / (y.norm(p=torch.inf, dim=-1) + self.eps)
        residues = residues.max(dim=1)[0]
        
        # TODO should this happen after the stepsize was updated?
        state['y'] = y * state['stepsize'] + state['y'] * (1 - state['stepsize'])

        # update patience and lowest residue
        decreased = residues < state['residues']
        state['residues'] = torch.where(decreased, residues,  state['residues'])
        state['patience'] = torch.where(decreased, self.decay_patience, state['patience']-1)

        # update damping factor, reset patience
        adapt = state['patience'] == 0
        state['patience'] += torch.where(adapt, self.decay_patience, 0)
        state['stepsize'] *= torch.where(adapt, self.stepsize_decay, 1).reshape(-1, 1, 1)

        # # find the inputs that have converged # TODO: necessary? or just use state['residues']?
        # state['converged'] = (residues < self.thresh)

        state['iter_idx'] += 1
        return state

    def cont(self, state:Dict[str, torch.Tensor]):
        '''optimizer continue: check convergence'''
        if state['iter_idx'] == 0: return True  # noqa: E701
        q = 1-self.outlier_quantile if self.training else 1
        max_iter = self.max_iter if self.training else 1000
        # max_iter = self.max_iter

        # TODO: state['residues'] are the lowest observed ones..
        # shouldn't we use the current ones in case they explode again?
        return (torch.quantile(state['residues'], q=q) >= self.thresh
                and state['stepsize'].max() >= self.stepsize_thresh
                and state['iter_idx'] < max_iter)
