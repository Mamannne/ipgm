import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. THE PHYSICS ENGINE (same as yours) ---

def temp_function(pos, vel):
    


class BouncingBallSim:
    def __init__(self, size=32, r=3, pos_start=None, vel_start=None):
        self.size = size
        self.r = r
        
        if pos_start is None:
            self.pos = np.random.rand(2) * (size - 2*r) + r
        else:
            self.pos = np.array(pos_start, dtype=float)
            
        if vel_start is None:
            self.vel = (np.random.rand(2) - 0.5) * 5
        else:
            self.vel = np.array(vel_start, dtype=float)
            
        self.gravity = -0.05
        
    def step(self):
        self.pos += self.vel
        self.vel[1] += self.gravity

        # bounces
        if (self.pos[0] + self.r >= self.size and self.vel[0] > 0) or \
           (self.pos[0] - self.r <= 0 and self.vel[0] < 0):
            self.vel[0] *= -0.7
        if (self.pos[1] + self.r >= self.size and self.vel[1] > 0) or \
           (self.pos[1] - self.r <= 0 and self.vel[1] < 0):
            self.vel[1] *= -0.7
        
        noisy_pos = self.pos + np.random.randn(2) * 0.5
        return noisy_pos, None, self.pos.copy()

def generate_sequences(n_sequences=1000, T=30):
    """
    Generate sequences of noisy observations x_{1:T}.
    x_t is 2D position with Gaussian noise.
    Returns: Tensor of shape (N, T, 2)
    """
    all_seqs = []
    for _ in range(n_sequences):
        sim = BouncingBallSim()
        seq = []
        for _ in range(T):
            obs, _, _ = sim.step()
            seq.append(obs)
        all_seqs.append(np.array(seq))
    x = torch.FloatTensor(np.stack(all_seqs, axis=0))  # (N, T, 2)
    return x



def reparameterize(mu, logvar):
    """
    mu, logvar: (B, T, D)
    returns: z ~ N(mu, diag(exp(logvar)))
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    """
    KL( N(mu_q, diag(sigma_q^2)) || N(mu_p, diag(sigma_p^2)) )
    All tensors shape: (B, T, D)
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * torch.sum(
        logvar_p - logvar_q + (var_q + (mu_q - mu_p)**2) / var_p - 1,
        dim=-1  # sum over D, keep (B,T)
    )




class DKF(nn.Module):
    def __init__(self, x_dim=2, z_dim=4, h_dim=64):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        
        # --- Generative model ---
        # Transition prior: z_{t-1} -> mean of z_t
        # Structured: new_pos = pos + vel; delta_v = f(z_{t-1}); new_vel = vel + delta_v
        self.transition_net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # delta_vx, delta_vy
        )
        # Global transition log-variance (diagonal)
        self.transition_logvar = nn.Parameter(torch.zeros(z_dim))
        
        # Initial prior p(z1) = N(0, I) (fixed)
        self.z1_mu = nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        self.z1_logvar = nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        
        # Emission: x_t | z_t ~ N(H z_t, R)
        # Emission: x_t | z_t ~ N(decoder(z_t), R)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, x_dim)
        )
        self.emission_logvar = nn.Parameter(torch.zeros(x_dim))
        
        # --- Inference model q(z_t | x_{1:T}) ---
        self.encoder_rnn = nn.GRU(
            input_size=x_dim,
            hidden_size=h_dim,
            batch_first=True,
            bidirectional=True
        )
        self.encoder_mlp = nn.Sequential(
            nn.Linear(2*h_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2*z_dim)  # outputs [mu, logvar]
        )
        
    def transition_mean(self, z_prev):
        """
        z_prev: (B, T-1, 4) or (B, 4)
        return: mean_z_t, same shape as z_prev
        """
        # split pos, vel
        pos = z_prev[..., :2]
        vel = z_prev[..., 2:]
        
        # normalization for stability (similar to your code)
        scale = torch.tensor([32.0, 32.0, 5.0, 5.0], device=z_prev.device)
        z_norm = z_prev / scale
        
        delta_v = self.transition_net(z_norm)
        new_vel = vel + delta_v
        new_pos = pos + vel
        
        return torch.cat([new_pos, new_vel], dim=-1)
    
    def emission_log_prob(self, x, z):
        """
        x: (B, T, x_dim)
        z: (B, T, z_dim)
        log p(x | z) with neural decoder
        """
        # decode mean
        B, T, _ = x.shape
        z_flat = z.reshape(B*T, self.z_dim)
        mu_flat = self.decoder(z_flat)              # (B*T, x_dim)
        mu = mu_flat.reshape(B, T, self.x_dim)      # (B, T, x_dim)
        
        logvar = self.emission_logvar              # (x_dim,)
        var = torch.exp(logvar)
        
        diff = x - mu
        log_prob = -0.5 * (
            torch.sum((diff**2) / var, dim=-1) +
            torch.sum(logvar) +
            self.x_dim * np.log(2 * np.e * np.pi)
        )  # (B, T)
        return log_prob

    
    def encoder(self, x):
        """
        x: (B, T, x_dim)
        returns: mu_q, logvar_q of shape (B, T, z_dim)
        """
        B, T, _ = x.shape
        h_seq, _ = self.encoder_rnn(x)  # (B, T, 2*h_dim)
        enc_out = self.encoder_mlp(h_seq)  # (B, T, 2*z_dim)
        mu_q, logvar_q = torch.chunk(enc_out, 2, dim=-1)
        return mu_q, logvar_q
    
    def forward(self, x):
        """
        Compute negative ELBO for batch of sequences.
        x: (B, T, 2)
        returns: loss (scalar), elbo (scalar)
        """
        B, T, _ = x.shape
        
        # 1) Inference: q(z_t | x_{1:T})
        mu_q, logvar_q = self.encoder(x)       # (B, T, z_dim)
        z_samples = reparameterize(mu_q, logvar_q)  # (B, T, z_dim)
        
        # 2) Generative: priors
        # z1 prior
        mu_p1 = self.z1_mu.view(1, 1, -1).expand(B, 1, -1)
        logvar_p1 = self.z1_logvar.view(1, 1, -1).expand(B, 1, -1)
        
        # compute transition priors for t >= 2
        z_prev = z_samples[:, :-1, :]            # (B, T-1, z_dim)
        mu_pt = self.transition_mean(z_prev)     # (B, T-1, z_dim)
        logvar_pt = self.transition_logvar.view(1,1,-1).expand(B, T-1, -1)
        
        # 3) Likelihood term
        log_px_given_z = self.emission_log_prob(x, z_samples)  # (B, T)
        recon_term = log_px_given_z.sum(dim=1)                 # (B,)
        
        # 4) KL terms
        # KL for t=1: q(z1|x) || p(z1)
        kl_1 = kl_diag_gaussians(mu_q[:, :1, :], logvar_q[:, :1, :], mu_p1, logvar_p1)  # (B,1)
        
        # KL for t>=2: q(z_t|x) || p(z_t|z_{t-1})
        kl_t = kl_diag_gaussians(
            mu_q[:, 1:, :], logvar_q[:, 1:, :], mu_pt, logvar_pt
        )  # (B, T-1)
        
        kl_term = kl_1.sum(dim=1) + kl_t.sum(dim=1)   # (B,)
        
        elbo = recon_term - kl_term   # (B,)
        loss = -elbo.mean()
        return loss, elbo.mean()
    
    @torch.no_grad()
    def infer_posterior_mean(self, x):
        """
        x: (1, T, 2) -> return posterior mean of z_t, shape (T, z_dim)
        """
        self.eval()
        mu_q, logvar_q = self.encoder(x)
        return mu_q.squeeze(0).cpu().numpy()
