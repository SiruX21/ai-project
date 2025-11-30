# Flappy Bird Reinforcement Learning

This project implements a Deep Q-Network (DQN) agent to play Flappy Bird using PyTorch.

## Project Structure

- `flappy.py` - Original Flappy Bird game
- `flappy_env.py` - RL environment wrapper (Gym-like interface)
- `dqn_agent.py` - DQN agent implementation with replay buffer and target network
- `train_dqn.py` - Standard training script (fixed number of episodes)
- `train_iterative.py` - **Iterative training script** (trains until near-perfect)
- `evaluate.py` - Evaluation and visualization script

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

**Option 1: Standard Training (Fixed Episodes)**
```bash
python train_dqn.py
```

The training script will:
- Train for 2000 episodes by default
- Save models periodically to `models/` directory
- Generate training curves plot (`training_curves.png`)
- Print progress every 10 episodes

**Option 2: Iterative Training (Until Near-Perfect) ⭐ Recommended**
```bash
python train_iterative.py
```

The iterative training script will:
- Train in iterations (200 episodes per iteration)
- Evaluate after each iteration
- Continue training until target performance is achieved
- Automatically save the best model
- Stop early if no improvement (patience mechanism)
- Generate detailed training curves

**Iterative Training Features:**
- **Target-based**: Stops when mean score ≥ 50 and max score ≥ 100
- **Best model tracking**: Always keeps the best performing model
- **Early stopping**: Stops if no improvement for 5 consecutive iterations
- **Checkpointing**: Saves models periodically
- **Progress tracking**: Shows detailed stats after each iteration

You can customize targets in `train_iterative.py`:
```python
agent, history = train_until_perfect(
    target_mean_score=50,      # Target average score
    target_max_score=100,       # Target max score
    episodes_per_iteration=200, # Episodes per iteration
    max_iterations=50,          # Maximum iterations
    patience=5,                 # Early stopping patience
)
```

### Evaluation

Evaluate a trained model (headless, no rendering):
```bash
python evaluate.py --model models/dqn_final.pth --episodes 20
```

Evaluate with rendering:
```bash
python evaluate.py --model models/dqn_final.pth --episodes 20 --render
```

Watch the agent play:
```bash
python evaluate.py --model models/dqn_final.pth --watch --games 5
```

## Environment Details

### State Representation

The state is a 5-dimensional vector:
- `bird_y`: Normalized bird y-position (0-1)
- `bird_velocity`: Normalized bird velocity
- `distance_to_next_pipe_x`: Normalized horizontal distance to next pipe
- `gap_top_y`: Normalized top of gap y-position
- `gap_bottom_y`: Normalized bottom of gap y-position

### Actions

- `0`: No flap (let gravity act)
- `1`: Flap (apply upward velocity)

### Rewards

- `+1.0`: Living reward (each frame alive)
- `+5.0`: Bonus for passing a pipe
- `-10.0`: Penalty for crashing

## DQN Architecture

- **Q-Network**: 3-layer MLP (128 → 128 → 64 → 2)
- **Replay Buffer**: Capacity 50,000 transitions
- **Target Network**: Updated every 1000 steps
- **Exploration**: ε-greedy with exponential decay (1.0 → 0.05 over 50k steps)
- **Learning Rate**: 1e-3 (Adam optimizer)
- **Discount Factor**: 0.99

## Training Hyperparameters

- Episodes: 2000
- Max steps per episode: 2000
- Batch size: 64
- Target network update frequency: 1000 steps
- Model save frequency: Every 500 episodes

## Results

After training, you should see:
- Training curves showing learning progress
- Models saved in `models/` directory
- Evaluation statistics (mean score, max score, etc.)

## Notes

- Training is done without rendering for speed
- Use `--render` flag during evaluation to visualize
- Models are saved in PyTorch format (`.pth` files)
- Training curves are automatically saved as `training_curves.png`

## CUDA/GPU Support

**Automatic CUDA Detection:**
- The code **automatically uses NVIDIA CUDA** if available for faster training
- Training script will detect and use your GPU automatically
- Evaluation script also uses CUDA by default if available
- You'll see a message indicating which GPU is being used when training starts

**Manual Control:**
- To force CPU in training: Modify `train_dqn.py` to set `device="cpu"`
- To force CPU in evaluation: Use `--device cpu` flag
- Example: `python evaluate.py --model models/dqn_final.pth --device cpu`

**Performance:**
- GPU training is significantly faster (often 5-10x speedup)
- All neural network operations run on GPU
- Experience replay buffer stays on CPU (more efficient for large buffers)

## Troubleshooting

**Pygame initialization errors on headless systems:**
- The environment initializes pygame in headless mode when `render=False`
- If you encounter display issues, try setting `SDL_VIDEODRIVER=dummy` environment variable

**CUDA not detected:**
- Ensure PyTorch was installed with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify NVIDIA drivers are installed: `nvidia-smi`

