import torch

class NonMarkovianSDE(torch.nn.Module):
    """
    Implementation of Neural SDE with non-Markovian diffusion term:
    dx_t = f_θ(x_t, t)dt + ξ_t dt
    dξ_t = -λξ_t dt + σdW_t
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, lambda_reversion=1.0, sigma=1.0, reverse=False):
        """
        Args:
            ode_drift: Neural network for drift term f_θ
            lambda_reversion: Mean reversion rate λ for the noise process
            sigma: Diffusion intensity σ
            reverse: Whether to reverse the time direction
        """
        super().__init__()
        self.drift = ode_drift
        self.reverse = reverse
        self.lambda_reversion = lambda_reversion
        self.sigma = sigma
        
    def augmented_state(self, y):
        """Split augmented state into position and noise components"""
        # First half of dimensions is x_t, second half is ξ_t
        dim = y.shape[-1] // 2
        x = y[..., :dim]
        xi = y[..., dim:]
        return x, xi

    def f(self, t, y, condition=None):
        """
        Drift function for the augmented system [x_t, ξ_t]
        Returns: [f_θ(x_t, t) + ξ_t, -λξ_t]
        """
        if self.reverse:
            t = 1 - t
                
        x, xi = self.augmented_state(y)
        
        # Prepare time tensor
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.view(-1, 1)
        
        # Compute neural network drift for x
        if condition is not None:
            # Make sure t matches batch size
            t_expanded = t if t.shape[0] == x.shape[0] else t.expand(x.shape[0], -1)
            nn_input = torch.cat([x, condition, t_expanded], dim=1)
        else:
            t_expanded = t if t.shape[0] == x.shape[0] else t.expand(x.shape[0], -1)
            nn_input = torch.cat([x, t_expanded], dim=1)
        
        f_theta = self.drift(nn_input)
        
        # Combined drift for x_t: f_θ(x_t, t) + ξ_t
        x_drift = f_theta + xi
        
        # Drift for ξ_t: -λξ_t
        xi_drift = -self.lambda_reversion * xi
                
        # Combine drifts for augmented system
        return torch.cat([x_drift, xi_drift], dim=-1)

    def g(self, t, y):
        """
        Diffusion function for the augmented system [x_t, ξ_t]
        Returns: [0, σ]
        """
        x, xi = self.augmented_state(y)
        
        # No direct noise in x_t equation
        x_diffusion = torch.zeros_like(x)
        
        # σ noise in ξ_t equation
        xi_diffusion = torch.ones_like(xi) * self.sigma
        
        return torch.cat([x_diffusion, xi_diffusion], dim=-1)

    def init_augmented_state(self, x0):
        """Initialize augmented state with x0 and ξ0"""
        batch_size = x0.shape[0]
        dim = x0.shape[1]
        
        # Initialize ξ0 from normal distribution
        xi0 = torch.randn(batch_size, dim, device=x0.device) * self.sigma
        
        # Return augmented state [x0, ξ0]
        return torch.cat([x0, xi0], dim=-1) 