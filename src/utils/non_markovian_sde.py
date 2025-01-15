import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model with the suggested hyperparameters
class NonMarkovianDriftNet(nn.Module):
    def __init__(self, dim, treatment_cond, memory, w, num_layers, activation_fn, dropout_rate):
        super().__init__()
        self.dim = dim
        self.memory = memory
        self.treatment_cond = treatment_cond
        
        self.indim = dim + treatment_cond + 1 + (dim * memory)
        self.out_dim = dim + 1
        
        layers = []
        layers.append(nn.Linear(self.indim, w))
        layers.append(activation_fn)
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(w, w))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(w, self.out_dim))
        
        self.net = nn.Sequential(*layers)

    def forward_train(self, x):
        return self.net(x)

    def forward(self, x):
        x1 = self.forward_train(x)
        x1_coord = x1[:, :self.dim]
        t = x[:, -1:]
        pred_time_till_t1 = x1[:, -1:]
        x_coord = x[:, :self.dim]
        vt = (x1_coord - x_coord)/(pred_time_till_t1 + 1e-6)
        return vt

class NonMarkovianNoiseNet(nn.Module):
    def __init__(self, dim, treatment_cond, memory, w, num_layers, activation_fn, dropout_rate):
        super().__init__()
        self.dim = dim
        self.memory = memory
        self.treatment_cond = treatment_cond
        
        self.indim = dim + treatment_cond + 1 + (dim * memory)
        self.out_dim = 1
        
        layers = []
        layers.append(nn.Linear(self.indim, w))
        layers.append(activation_fn)
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(w, w))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(w, self.out_dim))
        
        self.net = nn.Sequential(*layers)

    def forward_train(self, x):
        return self.net(x)

    def forward(self, x):
        return self.forward_train(x)
    
class NonMarkovianSDEModel(nn.Module):
    def __init__(self, treatment_cond, memory=3, dim=1, w=64, lr=1e-6, sigma=0.01, implementation="SDE", sde_noise=0.01, weight_decay=1e-4, num_layers=3, activation_fn=torch.nn.SELU(), dropout_rate=0.2, loss_fn=F.mse_loss):
        super().__init__()
        self.memory = memory
        self.dim = dim
        self.implementation = implementation
        self.sde_noise = sde_noise
        self.sigma = sigma
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        
        self.flow_model = NonMarkovianDriftNet(dim=dim, treatment_cond=treatment_cond, memory=memory, w=w, num_layers=num_layers, activation_fn=activation_fn, dropout_rate=dropout_rate)
        
        if implementation == "SDE":
            self.noise_model = NonMarkovianNoiseNet(dim=dim, treatment_cond=treatment_cond, memory=memory, w=w, num_layers=num_layers, activation_fn=activation_fn, dropout_rate=dropout_rate)

    def forward(self, x):
        return self.flow_model(x)

    def configure_optimizers(self):
        flow_optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        noise_optimizer = torch.optim.Adam(self.noise_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return flow_optimizer, noise_optimizer

    def __x_processing__(self, x0, x1, t0, t1):
        t = torch.rand(x0.shape[0], 1).to(x0.device)
        mu_t = x0 * (1 - t) + x1 * t
        data_t_diff = t1 - t0
        x = mu_t + self.sigma * torch.randn_like(x0)
        ut = (x1 - x0) / (data_t_diff + 1e-4)
        t_model = t * data_t_diff + t0
        futuretime = t1 - t_model
        
        memory = torch.zeros(x0.shape[0], self.dim * self.memory, device=x0.device)
        x = torch.cat([x, memory], dim=-1)
        
        return x, ut, t_model, futuretime, t

    def init_augmented_state(self, x0):
        batch_size = x0.shape[0]
        memory = torch.zeros(batch_size, self.dim * self.memory, device=x0.device)
        return torch.cat([x0, memory], dim=-1)

    def augmented_state(self, y):
        state = y[..., :self.dim]
        memory = y[..., self.dim:]
        return state, memory

    def f(self, t, y, condition=None):
        state, memory = self.augmented_state(y)
        
        if not torch.is_tensor(t):
            t = torch.tensor(t).to(state.device)
        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.view(-1, 1)
            
        t_expanded = t if t.shape[0] == state.shape[0] else t.expand(state.shape[0], -1)
        
        if condition is not None:
            nn_input = torch.cat([state, memory, condition, t_expanded], dim=1)
        else:
            nn_input = torch.cat([state, memory, t_expanded], dim=1)
        
        return self.flow_model(nn_input)

    def g(self, t, y):
        state, memory = self.augmented_state(y)
        state_diffusion = torch.zeros_like(state)
        memory_diffusion = torch.ones_like(memory) * self.sigma
        return torch.cat([state_diffusion, memory_diffusion], dim=-1)