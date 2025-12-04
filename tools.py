import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

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
    colors = ['red', 'blue', 'green']
    
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




def compare_models(models,consistent = True,sim = None):
    # 1. Setup Simulation (consistent=True to Force a specific bounce for consistency)
    if consistent:
        sim = BouncingBallSim(pos_start=[15, 15], vel_start=[2.0, 1.0]) 
    else:
        if sim is None:
            sim = BouncingBallSim()
    data = [sim.step() for _ in range(100)]
    observations = np.array([d[0] for d in data]) 
    ground_truth = np.array([d[2] for d in data]) 
    
    estimates = {}
    
    for model in models:
        # Check if Classical KF (PyKalman object)
        if hasattr(model, 'filter'):
            print("Running Classical KF...")
            model.initial_state_mean = [observations[0, 0], observations[0, 1], 0, 0]
            kf_mean, _ = model.filter(observations)
            estimates[model] = kf_mean
            
        else:
            # Deep KF Variants
            print(f"Running Deep Model: {type(model).__name__}...")
            dkf_est = []
            
            # --- FIX STARTS HERE ---
            # Frame 0: We know Position, guess Velocity=0
            # We must concatenate so it has shape (4,) matches the rest!
            obs0_padded = np.concatenate([observations[0], [0, 0]])
            dkf_est.append(obs0_padded) 
            # --- FIX ENDS HERE ---
            
            # Frame 1: We calculate velocity from diff
            init_vel = (observations[1] - observations[0])
            current_state = torch.FloatTensor(np.concatenate([observations[1], init_vel]))
            dkf_est.append(current_state.numpy()) 
            
            with torch.no_grad():
                # Loop starting from index 2
                for z in observations[2:]:
                    # A. PREDICT
                    if next(model.parameters()).is_cuda:
                        current_state = current_state.to('cuda')
                        
                    prediction = model(current_state.unsqueeze(0)).squeeze(0)       # (pos_estimate,vel_estimate) size = (4,1)
                    prediction = prediction.cpu() 
                    
                    # B. UPDATE (Coupled)
                    z_tensor = torch.FloatTensor(z)                                 # Observation (pos only) size = (2,1)
                    pred_pos = prediction[:2]
                    pred_vel = prediction[2:]
                    
                    residual = z_tensor - pred_pos
                    
                    corrected_pos = pred_pos + 0.2 * residual 
                    corrected_vel = pred_vel + 0.1 * residual
                    
                    current_state = torch.cat([corrected_pos, corrected_vel])
                    dkf_est.append(current_state.numpy())
            
            estimates[model] = np.array(dkf_est)
            
    return ground_truth, observations, estimates