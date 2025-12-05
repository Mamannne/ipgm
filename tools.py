import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import torch.nn.functional as F

# --- 1. THE PHYSICS ENGINE ---
class BouncingBallSim:
    def __init__(self, size=32, r=3, pos_start=None, vel_start=None):
        self.size = size
        self.r = r
        
        # If no position given, randomize it
        if pos_start is None:
            self.pos = np.random.rand(2) * (size - 2*r) + r
        else:
            self.pos = np.array(pos_start, dtype=float)
            
        # If no velocity given, randomize it
        if vel_start is None:
            self.vel = (np.random.rand(2) - 0.5) * 5
        else:
            self.vel = np.array(vel_start, dtype=float)
            
        self.gravity = -0.05
        
    def step(self):
        # 1. Physics (Linear Motion + Gravity)
        self.pos += self.vel
        self.vel[1] += self.gravity  # Gravity acts on Y
        
        # 2. Non-Linearity (The Bounce)
        # Hitting Right/Left Walls
        if (self.pos[0] + self.r >= self.size and self.vel[0] > 0) or (self.pos[0] - self.r <= 0 and self.vel[0] < 0):
            self.vel[0] *= -0.7 # Reverse and lose energy
        # Hitting Floor/Ceiling
        if (self.pos[1] + self.r >= self.size and self.vel[1] > 0) or (self.pos[1] - self.r <= 0 and self.vel[1] < 0):
            self.vel[1] *= -0.7
            
        # 3. Generate Observation
        noisy_pos = self.pos + np.random.randn(2) * 0.5
        
        # We return truth in index 2 to match your logic
        return noisy_pos, None, self.pos.copy()

def generate_balanced_training_data(n_samples=5000):
    """
    Generates training pairs (State_t -> State_t+1).
    Oversamples wall collisions to ensure the network learns the bounce.
    """
    sim = BouncingBallSim()
    inputs = []
    targets = []
    
    for _ in range(n_samples):
        # STRATEGY: 20% of data should be "Near Wall" to teach the bounce
        if np.random.rand() < 0.2:
            # Force position to be near the right wall (e.g., 29-32)
            sim.pos[0] = np.random.uniform(29, 32)
            sim.pos[1] = np.random.uniform(0, 32)
        else:
            sim.pos = np.random.rand(2) * 30 + 1
            
        sim.vel = (np.random.rand(2) - 0.5) * 3 
        
        state_t = np.concatenate([sim.pos, sim.vel])
        sim.step() # Step forward
        state_t_plus_1 = np.concatenate([sim.pos, sim.vel])
        
        inputs.append(state_t)
        targets.append(state_t_plus_1)
        
    return torch.FloatTensor(np.array(inputs)), torch.FloatTensor(np.array(targets))


# --- 2. THE DEEP MODEL ---
class TransitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 inputs (x, y, vx, vy) -> 4 outputs (next_x, next_y, next_vx, next_vy)
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
    def forward(self, x):
        # NORMALIZATION: Helps the network handle different scales
        # Pos ~ 32, Vel ~ 3. We scale them to be roughly ~1.0
        scale = torch.tensor([32.0, 32.0, 3.0, 3.0]).to(x.device)
        x_norm = x / scale
        return self.net(x_norm)
    
class StructuredTransitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # We only input Position & Velocity (4)
        # We only output Velocity Update (2) -> Acceleration/Force
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Outputting delta_vx, delta_vy
        )
        
    def forward(self, x):
        # x is [pos_x, pos_y, vel_x, vel_y]
        
        # 1. Separate Input
        pos = x[:, :2]
        vel = x[:, 2:]
        
        # 2. Normalization (Still important!)
        # Scale pos by 32, vel by 3
        x_norm = x / torch.tensor([32.0, 32.0, 3.0, 3.0]).to(x.device)
        
        # 3. Predict FORCES (Acceleration)
        # "delta_vel" represents gravity + bounce reaction
        delta_vel = self.net(x_norm) 
        
        # 4. Integrate Physics (The "Structure")
        # New Velocity = Old Velocity + Delta (Network prediction)
        new_vel = vel + delta_vel
        
        # New Position = Old Position + Old Velocity (Standard Physics)
        # We DON'T ask the network to predict this. We trust Newton.
        new_pos = pos + vel 
        
        # Recombine
        return torch.cat([new_pos, new_vel], dim=1)


# --- 3. ANIMATION HELPER ---
# In tools.py
def animate_trajectories(ground_truth, estimates=None, labels=None, dt=0.02):
    """
    Animates bouncing ball trajectories.
    ground_truth: (N, 2) array or tuple of arrays
    estimates: list of (N, 2) or (N, 4) arrays
    """
    def parse_input(seq):
        seq = np.array(seq)
        # FIX: Check if shape[1] is >= 2 (handles 4D state vectors)
        if seq.ndim == 2 and seq.shape[1] >= 2 and seq.shape[0] > 2: 
            return seq[:, 0], seq[:, 1] # Take only X and Y
        elif seq.shape[0] == 2: 
            return seq[0], seq[1]
        return seq 

    gt_x, gt_y = parse_input(ground_truth)
    
    est_data = []
    if estimates:
        if not isinstance(estimates, list): estimates = [estimates]
        for est in estimates:
            est_data.append(parse_input(est))
            
    if not labels: labels = [f"Est {i+1}" for i in range(len(est_data))]

    # Setup Plot
    # (Combine all Xs and Ys to find plot limits)
    all_x = np.concatenate([gt_x] + [e[0] for e in est_data])
    all_y = np.concatenate([gt_y] + [e[1] for e in est_data])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(np.min(all_x) - 1, np.max(all_x) + 1)
    ax.set_ylim(np.min(all_y) - 1, np.max(all_y) + 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Trajectory Playback")

    # Graphics
    ball_gt, = ax.plot([], [], 'o', color='black', markersize=8, label='Ground Truth')
    line_gt, = ax.plot([], [], '--', color='grey', alpha=0.5)
    
    balls_est = []
    lines_est = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    
    for i, (est_x, est_y) in enumerate(est_data):
        c = colors[i % len(colors)]
        b, = ax.plot([], [], 'o', color=c, markersize=8, label=labels[i])
        l, = ax.plot([], [], '-', color=c, alpha=0.5, linewidth=1)
        balls_est.append(b)
        lines_est.append(l)

    ax.legend()

    def update(frame):
        # Update GT
        ball_gt.set_data([gt_x[frame]], [gt_y[frame]]) 
        line_gt.set_data(gt_x[:frame], gt_y[:frame])
        
        # Update Estimates
        for i, (est_x, est_y) in enumerate(est_data):
            if frame < len(est_x):
                balls_est[i].set_data([est_x[frame]], [est_y[frame]])
                lines_est[i].set_data(est_x[:frame], est_y[:frame])
                
        return [ball_gt, line_gt] + balls_est + lines_est

    frames = len(gt_x)
    anim = FuncAnimation(fig, update, frames=frames, interval=dt*1000, blit=True)
    plt.close() 
    return HTML(anim.to_jshtml())




def compare_models(models, consistent=True, sim=None):
    # 1. Setup Simulation
    if consistent:
        # Force a specific bounce for consistency
        sim = BouncingBallSim(pos_start=[15, 15], vel_start=[2.0, 1.0]) 
    else:
        if sim is None:
            sim = BouncingBallSim()
            
    # Generate 100 steps
    data = [sim.step() for _ in range(100)]
    observations = np.array([d[0] for d in data]) # (100, 2)
    ground_truth = np.array([d[2] for d in data]) # (100, 2)
    
    estimates = {}
    
    for model in models:
        # --- CAS 1: KVAE (Nouveau) ---
        # On vérifie le nom de la classe pour éviter les erreurs d'import
        if type(model).__name__ == 'KVAE':
            print("Running Deep Model: KVAE (Smoothing)...")
            
            # Le KVAE prend toute la séquence d'un coup (Batch=1, T=100, Dim=2)
            obs_tensor = torch.FloatTensor(observations).unsqueeze(0) 
            
            # Gestion GPU
            device = next(model.parameters()).device
            obs_tensor = obs_tensor.to(device)
                
            with torch.no_grad():
                # model.smooth retourne la reconstruction (1, T, 2)
                recon = model.smooth(obs_tensor)
                
            # On stocke le résultat en numpy (T, 2)
            estimates[model] = recon.squeeze(0).cpu().numpy()

        # --- CAS 2: Filtre de Kalman Classique (PyKalman) ---
        elif hasattr(model, 'filter'):
            print("Running Classical KF...")
            # Init naive : on suppose vitesse nulle au début
            model.initial_state_mean = [observations[0, 0], observations[0, 1], 0, 0]
            kf_mean, _ = model.filter(observations)
            estimates[model] = kf_mean
            
        # --- CAS 3: Modèles Deep "Pas à Pas" (DKF, SKF, TransitionNet) ---
        else:
            print(f"Running Deep Model: {type(model).__name__} (Step-by-Step)...")
            dkf_est = []
            
            # Frame 0: On connait la Position, on devine Vitesse=0
            obs0_padded = np.concatenate([observations[0], [0, 0]])
            dkf_est.append(obs0_padded) 
            
            # Frame 1: On calcule la vitesse initiale par différence finie
            init_vel = (observations[1] - observations[0])
            # État courant (4,) : [px, py, vx, vy]
            current_state = torch.FloatTensor(np.concatenate([observations[1], init_vel]))
            dkf_est.append(current_state.numpy()) 
            
            # On vérifie si le modèle est sur GPU
            is_cuda = next(model.parameters()).is_cuda
            
            with torch.no_grad():
                # Boucle à partir de l'image 2
                for z in observations[2:]:
                    # A. PREDICT
                    if is_cuda:
                        current_state = current_state.to('cuda')
                        
                    # Le modèle prédit la prochaine étape à partir de l'état estimé précédent
                    prediction = model(current_state.unsqueeze(0)).squeeze(0) 
                    prediction = prediction.cpu() 
                    
                    # B. UPDATE (Filtre "Manuel")
                    z_tensor = torch.FloatTensor(z) # Observation actuelle (pos seulement)
                    pred_pos = prediction[:2]
                    pred_vel = prediction[2:]
                    
                    # Calcul du résidu (Erreur de prédiction)
                    residual = z_tensor - pred_pos
                    
                    # Correction simple (Gain fixe)
                    # On corrige la position et un peu la vitesse
                    corrected_pos = pred_pos + 0.2 * residual 
                    corrected_vel = pred_vel + 0.1 * residual
                    
                    current_state = torch.cat([corrected_pos, corrected_vel])
                    dkf_est.append(current_state.numpy())
            
            estimates[model] = np.array(dkf_est)
            
    return ground_truth, observations, estimates



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KVAE(nn.Module):
    def __init__(self, x_dim=2, a_dim=2, z_dim=4, K=3, hidden_dim=64, scale=32.0):
        """
        KVAE: Kalman Variational Auto-Encoder
        
        Args:
            x_dim: Dimension de l'observation (2 pour x,y)
            a_dim: Dimension de l'objet latent (disentangled representation)
            z_dim: Dimension de l'état dynamique (LGSSM state)
            K: Nombre d'experts dynamiques (pour gérer les rebonds)
            scale: Facteur d'échelle pour normaliser les entrées (ex: 32.0 pour la boite)
        """
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.K = K
        
        # On enregistre l'échelle comme une constante du modèle
        self.register_buffer('scale', torch.tensor(float(scale)))

        # --- 1. VAE (OBSERVATION MODEL) ---
        # Encoder : x (brut) -> a (latent)
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * a_dim) # [mu, logvar]
        )
        
        # Decoder : a (latent) -> x (brut)
        self.decoder = nn.Sequential(
            nn.Linear(a_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, x_dim)
        )
        
        # Bruit d'observation (fixé bas pour forcer la précision)
        # log_sigma = -2.0 => sigma approx 0.13 pixel (une fois normalisé)
        self.log_scale_x = nn.Parameter(torch.tensor(-2.0), requires_grad=False)

        # --- 2. DYNAMICS PARAMETER NETWORK (LSTM) ---
        # Prédit le poids des experts (alpha) à partir de l'histoire de 'a'
        self.lstm = nn.LSTM(a_dim, hidden_dim, batch_first=True)
        self.lstm_to_alpha = nn.Linear(hidden_dim, K)

        # --- 3. LGSSM PARAMETERS (EXPERTS) ---
        # Matrices de transition (A) et d'émission (C) pour chaque expert k
        # Init: A proche de l'identité (mouvement fluide), C aléatoire petit
        self.A_k = nn.Parameter(torch.randn(K, z_dim, z_dim) * 0.05 + torch.eye(z_dim).unsqueeze(0))
        self.B_k = nn.Parameter(torch.zeros(K, z_dim))
        self.C_k = nn.Parameter(torch.randn(K, a_dim, z_dim) * 0.1)
        self.D_k = nn.Parameter(torch.zeros(K, a_dim))
        
        # Bruit de processus (Q) et d'état (R) - Appris
        self.Q_logvar = nn.Parameter(torch.zeros(z_dim) - 3.0) 
        self.R_logvar = nn.Parameter(torch.zeros(a_dim) - 3.0)
        
        # État initial p(z_0)
        self.z0_mu = nn.Parameter(torch.zeros(z_dim))
        self.z0_logvar = nn.Parameter(torch.zeros(z_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_x(self, x):
        """ Encode x en a avec normalisation interne """
        # x: (B, T, x_dim) dans [0, scale]
        # Normalisation vers [0, 1] pour la stabilité du réseau
        x_norm = x / self.scale
        
        B, T, _ = x_norm.shape
        flat_x = x_norm.contiguous().view(B*T, -1)
        out = self.encoder(flat_x)
        mu, logvar = torch.chunk(out, 2, dim=-1)
        return mu.view(B, T, -1), logvar.view(B, T, -1)

    def decode_a(self, a):
        """ Decode a en x avec dénormalisation """
        # a: (B, T, a_dim)
        B, T, _ = a.shape
        flat_a = a.contiguous().view(B*T, -1)
        out_norm = self.decoder(flat_a) # Sortie dans [0, 1] environ
        
        # Dénormalisation vers [0, scale]
        out_scaled = out_norm.view(B, T, -1) * self.scale
        return out_scaled

    def get_mixture_parameters(self, alpha):
        """ Mélange les matrices des experts selon les poids alpha """
        # alpha: (B, T, K)
        # A_k: (K, z, z) -> A_t: (B, T, z, z)
        A_t = torch.einsum('btk,kzy->btzy', alpha, self.A_k)
        B_t = torch.einsum('btk,kz->btz', alpha, self.B_k)
        C_t = torch.einsum('btk,kaz->btaz', alpha, self.C_k)
        D_t = torch.einsum('btk,ka->bta', alpha, self.D_k)
        return A_t, B_t, C_t, D_t

    def kalman_filter(self, a_obs, A, B_mat, C, D):
        """ Filtre de Kalman Différentiable (Calcul de la Vraisemblance) """
        B_size, T, _ = a_obs.shape
        device = a_obs.device
        
        mu = self.z0_mu.unsqueeze(0).expand(B_size, -1)
        Sigma = torch.diag(torch.exp(self.z0_logvar)).unsqueeze(0).expand(B_size, -1, -1)
        Q = torch.diag(torch.exp(self.Q_logvar)).unsqueeze(0).expand(B_size, -1, -1)
        R = torch.diag(torch.exp(self.R_logvar)).unsqueeze(0).expand(B_size, -1, -1)
        
        log_likelihood = 0.0
        
        for t in range(T):
            # 1. Prediction
            mu_pred = torch.bmm(A[:, t], mu.unsqueeze(-1)).squeeze(-1) + B_mat[:, t]
            Sigma_pred = torch.bmm(torch.bmm(A[:, t], Sigma), A[:, t].transpose(1, 2)) + Q
            
            # 2. Update / Correction
            # Projection dans l'espace 'a'
            y_pred = torch.bmm(C[:, t], mu_pred.unsqueeze(-1)).squeeze(-1) + D[:, t]
            
            # Résidu
            r_t = a_obs[:, t] - y_pred
            S_t = torch.bmm(torch.bmm(C[:, t], Sigma_pred), C[:, t].transpose(1, 2)) + R
            
            # Gain de Kalman
            S_inv = torch.inverse(S_t)
            K_t = torch.bmm(torch.bmm(Sigma_pred, C[:, t].transpose(1, 2)), S_inv)
            
            # Mise à jour État
            mu = mu_pred + torch.bmm(K_t, r_t.unsqueeze(-1)).squeeze(-1)
            I = torch.eye(self.z_dim, device=device).unsqueeze(0).expand(B_size, -1, -1)
            Sigma = torch.bmm(I - torch.bmm(K_t, C[:, t]), Sigma_pred)
            
            # 3. Log-Likelihood
            log_det_S = torch.logdet(S_t)
            quad_term = torch.sum(r_t.unsqueeze(1).bmm(S_inv) * r_t.unsqueeze(1), dim=-1).squeeze()
            ll_step = -0.5 * (log_det_S + quad_term + self.a_dim * np.log(2 * np.pi))
            log_likelihood += ll_step
            
        return log_likelihood

    def forward(self, x):
        """ Training Step: Retourne la Loss (Negative ELBO) """
        B, T, _ = x.shape
        
        # 1. Encodage x -> a
        mu_a, logvar_a = self.encode_x(x)
        a_sample = self.reparameterize(mu_a, logvar_a)
        
        # 2. LSTM (Cerveau) : a -> alpha (Choix des experts)
        # Shift causal: l'input t sert à prédire t+1
        lstm_input = torch.cat([torch.zeros(B, 1, self.a_dim, device=x.device), a_sample[:, :-1, :]], dim=1)
        lstm_out, _ = self.lstm(lstm_input)
        alpha = F.softmax(self.lstm_to_alpha(lstm_out), dim=-1)
        
        # 3. Paramètres Dynamiques
        A_t, B_t, C_t, D_t = self.get_mixture_parameters(alpha)
        
        # 4. Filtre de Kalman (Vraisemblance dynamique)
        log_p_a_given_alpha = self.kalman_filter(a_sample, A_t, B_t, C_t, D_t)
        
        # 5. Reconstruction x -> x_rec
        x_recon = self.decode_a(a_sample)
        
        # 6. Loss Calculation
        # Reconstruction (MSE pondérée par le bruit sigma_x)
        # Note: On divise par scale^2 implicitement si on considère l'erreur normalisée, 
        # mais ici on calcule l'erreur brute.
        recon_mse = torch.sum((x - x_recon)**2, dim=[1,2])
        var_x = torch.exp(self.log_scale_x)**2 
        # Pour que la loss soit cohérente avec les grandes valeurs (0-32), on adapte le terme
        recon_loss = 0.5 * recon_mse / var_x 
        
        # Entropie (Régularisation)
        entropy_q_a = 0.5 * torch.sum(1 + logvar_a, dim=[1,2])
        
        # ELBO = log p(x|a) + log p(a) - log q(a|x)
        elbo = -recon_loss + log_p_a_given_alpha + entropy_q_a
        
        return -elbo.mean(), recon_loss.mean(), -log_p_a_given_alpha.mean()

    @torch.no_grad()
    def smooth(self, x):
        """ Inference Step: Lissage de trajectoire """
        # 1. Encodage
        a_seq, _ = self.encode_x(x)
        B, T, _ = a_seq.shape
        device = x.device
        
        # 2. LSTM
        lstm_in = torch.cat([torch.zeros(B, 1, self.a_dim, device=device), a_seq[:, :-1, :]], dim=1)
        lstm_out, _ = self.lstm(lstm_in)
        alpha = F.softmax(self.lstm_to_alpha(lstm_out), dim=-1)
        
        # 3. Paramètres
        A, B_mat, C, D = self.get_mixture_parameters(alpha)
        
        # 4. Filtrage de Kalman sur la moyenne (Approximation du lissage)
        mu = self.z0_mu.unsqueeze(0).expand(B, -1)
        Sigma = torch.diag(torch.exp(self.z0_logvar)).unsqueeze(0).expand(B, -1, -1)
        Q = torch.diag(torch.exp(self.Q_logvar)).unsqueeze(0).expand(B, -1, -1)
        R = torch.diag(torch.exp(self.R_logvar)).unsqueeze(0).expand(B, -1, -1)
        
        smoothed_a = []
        
        for t in range(T):
            # Predict
            mu_pred = torch.bmm(A[:, t], mu.unsqueeze(-1)).squeeze(-1) + B_mat[:, t]
            Sigma_pred = torch.bmm(torch.bmm(A[:, t], Sigma), A[:, t].transpose(1, 2)) + Q
            
            # Update
            y_pred = torch.bmm(C[:, t], mu_pred.unsqueeze(-1)).squeeze(-1) + D[:, t]
            r = a_seq[:, t] - y_pred
            S = torch.bmm(torch.bmm(C[:, t], Sigma_pred), C[:, t].transpose(1, 2)) + R
            K_gain = torch.bmm(torch.bmm(Sigma_pred, C[:, t].transpose(1, 2)), torch.inverse(S))
            
            mu = mu_pred + torch.bmm(K_gain, r.unsqueeze(-1)).squeeze(-1)
            I = torch.eye(self.z_dim, device=device).unsqueeze(0).expand(B, -1, -1)
            Sigma = torch.bmm(I - torch.bmm(K_gain, C[:, t]), Sigma_pred)
            
            # Projection vers 'a' (lissé)
            a_smooth_t = torch.bmm(C[:, t], mu.unsqueeze(-1)).squeeze(-1) + D[:, t]
            smoothed_a.append(a_smooth_t)
            
        smoothed_a = torch.stack(smoothed_a, dim=1)
        
        # 5. Décodage
        return self.decode_a(smoothed_a)
    
def generate_trajectory_data(n_sequences=1000, T=50):
    """
    Generates sequences of shape (N, T, 2) for KVAE training using the existing BouncingBallSim.
    """
    sequences = []
    for _ in range(n_sequences):
        sim = BouncingBallSim() # Uses your existing class
        seq = []
        for _ in range(T):
            obs, _, _ = sim.step()
            seq.append(obs)
        sequences.append(seq)
    return torch.FloatTensor(np.stack(sequences, axis=0))