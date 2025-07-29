# ========== FIXED SAM OPTIMIZER IMPLEMENTATION ==========
import torch

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer for better generalization - FIXED VERSION"""
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """First step: compute and apply perturbation"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                
                # Initialize state if not exists
                if p not in self.state:
                    self.state[p] = {}
                
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Second step: apply actual parameter update"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # FIX: Check if e_w exists before accessing it
                if p in self.state and "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        """Combined step function"""
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    
    def _grad_norm(self):
        """Compute gradient norm"""
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        
        # Collect gradients
        grads = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]
        
        # Handle case when no gradients exist
        if len(grads) == 0:
            return torch.tensor(0.0, device=shared_device)
        
        norm = torch.norm(torch.stack(grads), dim=0).to(shared_device)
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
