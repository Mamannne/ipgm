import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. THE PHYSICS ENGINE (same as yours) ---



class BouncingBallSim:
    def __init__(self, size=16, r=3, pos_start=None, vel_start=None, omega=0.5,
                 temp_noise=0.1):
        self.size = size
        self.r = r
        self.omega = omega
        self.temp_noise = temp_noise
        
        if pos_start is None:
            self.pos = np.random.rand(2) * (size - 2*r) + r
        else:
            self.pos = np.array(pos_start, dtype=float)
            
        if vel_start is None:
            self.vel = (np.random.rand(2) - 0.5) * 10
        else:
            self.vel = np.array(vel_start, dtype=float)
            
        self.gravity = -0.1

    def compute_temperature(self):
        pos_norm = np.linalg.norm(self.pos)
        vel_norm = np.linalg.norm(self.vel)
        T = 0.5 * pos_norm**2 + np.sin(self.omega * vel_norm)
        return T
        
    def step(self):
        # 1. dynamics
        self.pos += self.vel
        self.vel[1] += self.gravity

        # 2. temperature-dependent drag (if you like)
        T_true = self.compute_temperature()
        drag = 0.01 * np.tanh(T_true)
        self.vel *= (1.0 - drag)

        # 3. bounces
        if (self.pos[0] + self.r >= self.size and self.vel[0] > 0) or \
           (self.pos[0] - self.r <= 0 and self.vel[0] < 0):
            self.vel[0] *= -0.7
        if (self.pos[1] + self.r >= self.size and self.vel[1] > 0) or \
           (self.pos[1] - self.r <= 0 and self.vel[1] < 0):
            self.vel[1] *= -0.7
        
        # 4. noisy observations
        noisy_pos = self.pos + np.random.randn(2) * 0.5
        noisy_T   = T_true + np.random.randn() * self.temp_noise

        obs = np.concatenate([noisy_pos, [noisy_T]])                 # (3,)
        gt_state = np.concatenate([self.pos.copy(), self.vel.copy(), [T_true]])  # (5,)

        return obs, None, gt_state



def generate_sequences(n_sequences=1000, T=30):
    """
    Generate sequences of noisy observations [x, y, T].
    Returns: Tensor of shape (N, T, 3)
    """
    all_seqs = []
    for _ in range(n_sequences):
        sim = BouncingBallSim()
        seq = []
        for _ in range(T):
            obs, _, _ = sim.step()   # obs is now (3,)
            seq.append(obs)
        all_seqs.append(np.array(seq))
    x = torch.FloatTensor(np.stack(all_seqs, axis=0))  # (N, T, 3)
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




class ClassicalKF:
    """
    Linear Kalman Filter for state z = [x, y, vx, vy].
    Observations are noisy positions [x, y].
    """
    def __init__(self, dt=1.0,
                 process_var_pos=1e-3,
                 process_var_vel=1e-2,
                 meas_var=0.25):
        """
        dt: time step
        process_var_pos: variance for position-related process noise
        process_var_vel: variance for velocity-related process noise
        meas_var: variance of measurement noise on x,y

        All covariances are diagonal in this simple model.
        """
        self.dt = dt

        # State z_t = [x, y, vx, vy]^T
        # Constant-velocity (CV) model:
        #
        # x_{t+1}  = x_t  + vx_t * dt
        # y_{t+1}  = y_t  + vy_t * dt
        # vx_{t+1} = vx_t
        # vy_{t+1} = vy_t
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

        # Observation: we only use positions [x, y]
        # z = [x, y, vx, vy]  ->  y = C z = [x, y]
        self.C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Process noise covariance Q (diagonal here)
        self.Q = np.diag([
            process_var_pos,
            process_var_pos,
            process_var_vel,
            process_var_vel
        ]).astype(float)

        # Measurement noise covariance R
        self.R = np.diag([meas_var, meas_var]).astype(float)

        # These will be set externally before running filter()
        self.initial_state_mean = np.zeros(4, dtype=float)
        self.initial_state_covariance = np.eye(4, dtype=float)

    def filter(self, observations):
        """
        Run forward Kalman filter.

        observations: array of shape (T, 2) with [x_obs, y_obs].

        Returns:
          means: (T, 4)   filtered state means
          covs:  (T, 4,4) filtered state covariances
        """
        observations = np.asarray(observations)
        T_steps = observations.shape[0]
        z_dim = 4

        means = np.zeros((T_steps, z_dim), dtype=float)
        covs = np.zeros((T_steps, z_dim, z_dim), dtype=float)

        x = self.initial_state_mean.copy()
        P = self.initial_state_covariance.copy()

        I = np.eye(z_dim, dtype=float)

        for t in range(T_steps):
            # 1) Predict
            x_pred = self.A @ x
            P_pred = self.A @ P @ self.A.T + self.Q

            # 2) Update with observation y_t
            y_t = observations[t]                      # (2,)
            innov = y_t - (self.C @ x_pred)           # innovation

            S = self.C @ P_pred @ self.C.T + self.R   # (2,2)
            K = P_pred @ self.C.T @ np.linalg.inv(S)  # (4,2)

            x = x_pred + K @ innov                    # (4,)
            P = (I - K @ self.C) @ P_pred             # (4,4)

            means[t] = x
            covs[t] = P

        return means, covs


class DKF(nn.Module):
    def __init__(self, x_dim=3, z_dim=4, h_dim=64):
        super().__init__()
        self.x_dim = x_dim   # 3: [x,y,T]
        self.z_dim = z_dim   # 4: [x,y,vx,vy]
        
        # --- Transition: z_{t-1} -> z_t ---
        self.transition_net = nn.Sequential(
            nn.Linear(self.z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Δv_x, Δv_y
        )
        self.transition_logvar = nn.Parameter(torch.zeros(self.z_dim))
        
        self.z1_mu = nn.Parameter(torch.zeros(self.z_dim), requires_grad=False)
        self.z1_logvar = nn.Parameter(torch.zeros(self.z_dim), requires_grad=False)
        
        # --- Decoder: x | z ---
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.x_dim)  # outputs [x_hat, y_hat, T_hat]
        )
        self.emission_logvar = nn.Parameter(torch.zeros(self.x_dim))
        
        # --- Encoder q(z_t | x_{1:T}) ---
        self.encoder_rnn = nn.GRU(
            input_size=self.x_dim,   # now 3
            hidden_size=h_dim,
            batch_first=True,
            bidirectional=True
        )
        self.encoder_mlp = nn.Sequential(
            nn.Linear(2*h_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2*self.z_dim)  # mean+logvar for 4D z
        )
        
    def transition_mean(self, z_prev):
        """
        z_prev: (B, T-1, 4)
        """
        pos = z_prev[..., :2]      # (B, T-1, 2)
        vel = z_prev[..., 2:4]     # (B, T-1, 2)
        
        scale = torch.tensor(
            [32.0, 32.0, 5.0, 5.0],
            device=z_prev.device
        ).view(1, 1, self.z_dim)   # (1,1,4)
        
        z_norm = z_prev / scale

        delta_v = self.transition_net(z_norm)  # (B, T-1, 2)
        new_vel = vel + delta_v
        new_pos = pos + vel

        return torch.cat([new_pos, new_vel], dim=-1)  # (B, T-1, 4)

    def emission_log_prob(self, x, z):
        B, T, _ = x.shape
        z_flat = z.reshape(B*T, self.z_dim)
        mu_flat = self.decoder(z_flat)              # (B*T, 3)
        mu = mu_flat.reshape(B, T, self.x_dim)      # (B, T, 3)
        
        logvar = self.emission_logvar              # (3,)
        var = torch.exp(logvar)
        
        diff = x - mu
        log_prob = -0.5 * (
            torch.sum((diff**2) / var, dim=-1) +
            torch.sum(logvar) +
            self.x_dim * np.log(2 * np.pi)
        )
        return log_prob

    # encoder() and forward() stay the same, they already use self.x_dim / self.z_dim


    def encoder(self, x):
        B, T, _ = x.shape
        h_seq, _ = self.encoder_rnn(x)         # (B, T, 2*h_dim)
        enc_out = self.encoder_mlp(h_seq)      # (B, T, 2*z_dim)
        mu_q, logvar_q = torch.chunk(enc_out, 2, dim=-1)
        return mu_q, logvar_q
    
    def forward(self, x):
        B, T, _ = x.shape
        
        # q(z|x)
        mu_q, logvar_q = self.encoder(x)
        z_samples = reparameterize(mu_q, logvar_q)
        
        # priors
        mu_p1 = self.z1_mu.view(1, 1, -1).expand(B, 1, -1)
        logvar_p1 = self.z1_logvar.view(1, 1, -1).expand(B, 1, -1)
        
        z_prev = z_samples[:, :-1, :]               # (B, T-1, z_dim)
        mu_pt = self.transition_mean(z_prev)        # (B, T-1, z_dim)
        logvar_pt = self.transition_logvar.view(1, 1, -1).expand(B, T-1, -1)
        
        # likelihood
        log_px_given_z = self.emission_log_prob(x, z_samples)  # (B, T)
        recon_term = log_px_given_z.sum(dim=1)                 # (B,)
        
        # KL terms
        kl_1 = kl_diag_gaussians(
            mu_q[:, :1, :], logvar_q[:, :1, :],
            mu_p1, logvar_p1
        )  # (B,1)
        
        kl_t = kl_diag_gaussians(
            mu_q[:, 1:, :], logvar_q[:, 1:, :],
            mu_pt, logvar_pt
        )  # (B, T-1)
        
        kl_term = kl_1.sum(dim=1) + kl_t.sum(dim=1)
        elbo = recon_term - kl_term
        loss = -elbo.mean()
        return loss, elbo.mean()
    
    @torch.no_grad()
    def infer_posterior_mean(self, x):
        self.eval()
        mu_q, logvar_q = self.encoder(x)
        return mu_q.squeeze(0).cpu().numpy()
