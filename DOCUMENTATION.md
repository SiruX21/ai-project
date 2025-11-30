# Flappy Bird Reinforcement Learning - Technical Documentation

## Overview

This project implements a Deep Q-Network (DQN) agent to play Flappy Bird using PyTorch. The agent learns to play the game through reinforcement learning, where it receives rewards for staying alive and passing pipes, and penalties for crashing.

## Architecture

### 1. Environment (`flappy_env.py`)

The `FlappyBirdEnv` class wraps the Flappy Bird game into a Gym-like reinforcement learning environment.

#### State Representation (5-dimensional vector)

The agent observes a compact state vector instead of raw pixels:

1. **bird_y**: Normalized bird y-position (0-1, where 0 is top, 1 is bottom)
2. **bird_velocity**: Normalized bird vertical velocity
3. **distance_to_next_pipe_x**: Normalized horizontal distance to the next pipe (0-1)
4. **gap_top_y**: Normalized y-position of the top of the gap (0-1)
5. **gap_bottom_y**: Normalized y-position of the bottom of the gap (0-1)

**Why this representation?**
- Low-dimensional (5D) vs. high-dimensional pixels (400x600x3 = 720,000D)
- Contains all necessary information for decision-making
- Faster training and inference
- No need for convolutional neural networks

#### Actions

- **Action 0**: No flap (let gravity act)
- **Action 1**: Flap (apply upward velocity)

#### Reward Structure

The reward function is carefully designed to encourage desired behavior:

- **Living reward**: +0.1 per frame (encourages survival)
- **Pipe passing bonus**: +10.0 (strong incentive to pass pipes)
- **Distance bonus**: +0.05 × (1 - normalized_distance) (encourages approaching pipes)
- **Crash penalty**: -100.0 (strong discouragement from crashing)

**Reward shaping rationale:**
- Small living reward prevents the agent from learning to just crash immediately
- Large pipe passing bonus creates a clear learning signal
- Crash penalty is large enough to discourage unsafe behavior
- Distance bonus provides intermediate feedback

### 2. DQN Agent (`dqn_agent.py`)

#### Q-Network Architecture

```
Input (5D state) 
  ↓
Linear(5 → 128) + ReLU
  ↓
Linear(128 → 128) + ReLU
  ↓
Linear(128 → 64) + ReLU
  ↓
Linear(64 → 2) [Q-values for each action]
```

**Why this architecture?**
- Fully connected layers are sufficient for low-dimensional state
- 128-128-64 provides enough capacity without overfitting
- ReLU activations enable non-linear decision boundaries
- Outputs Q-values for both actions (no flap, flap)

#### Key Components

**1. Experience Replay Buffer**
- **Capacity**: 50,000 transitions
- **Purpose**: Breaks correlation between consecutive experiences
- **How it works**: Stores (state, action, reward, next_state, done) tuples
- **Sampling**: Randomly samples batches for training

**2. Target Network**
- **Purpose**: Stabilizes training by providing stable Q-value targets
- **Update frequency**: Every 500 steps
- **How it works**: Maintains a separate network with frozen weights, periodically updated from main network

**3. Epsilon-Greedy Exploration**
- **Initial epsilon**: 1.0 (100% random actions)
- **Final epsilon**: 0.01 (1% random actions)
- **Decay**: Exponential decay over 20,000 steps
- **Purpose**: Balances exploration (trying new actions) vs. exploitation (using learned policy)

#### Training Process

**DQN Algorithm (Mnih et al., 2015):**

1. **Collect experience**: Agent interacts with environment, stores transitions in replay buffer
2. **Sample batch**: Randomly sample 64 transitions from buffer
3. **Compute targets**: 
   - For non-terminal: `target = reward + γ × max(Q_target(next_state))`
   - For terminal: `target = reward`
4. **Compute loss**: Mean squared error between Q(s,a) and target
5. **Update network**: Backpropagate and update Q-network weights
6. **Update target**: Periodically copy Q-network weights to target network

**Hyperparameters:**
- Learning rate: 5e-4 (Adam optimizer)
- Discount factor (γ): 0.99
- Batch size: 64
- Target update frequency: 500 steps
- Gradient clipping: Max norm 10

### 3. Training Scripts

#### `train_iterative.py` - Iterative Training

**Process:**
1. Train for N episodes (default: 200)
2. Evaluate agent performance
3. Check if targets achieved
4. Continue training if not
5. Stop if no improvement for patience iterations

**Features:**
- Automatic best model tracking
- Early stopping to prevent overfitting
- Progress reporting every 20 episodes
- Live demonstrations during training
- Comprehensive final evaluation

**Targets:**
- Mean score: 200+ pipes
- Max score: 500+ pipes
- Continues training even after targets to push for maximum performance

#### `train_dqn.py` - Standard Training

Fixed number of episodes training (2000 by default). Useful for baseline comparisons.

### 4. Evaluation (`evaluate.py`)

**Modes:**
- **Headless evaluation**: Fast evaluation without rendering
- **Rendered evaluation**: Visual evaluation with pygame window
- **Watch mode**: Live demonstration of agent playing

**Metrics:**
- Mean score (pipes passed)
- Standard deviation
- Min/Max scores
- Percentiles (median, 75th, 90th)

## Learning Process

### Phase 1: Random Exploration (Episodes 0-500)
- Agent mostly takes random actions (high epsilon)
- Learns basic game mechanics
- Replay buffer fills up
- Q-values start to differentiate between actions

### Phase 2: Early Learning (Episodes 500-2000)
- Epsilon decreases, more exploitation
- Agent learns to avoid immediate crashes
- Starts to understand pipe positioning
- Occasional pipe passes

### Phase 3: Skill Development (Episodes 2000-5000)
- Consistent pipe passing
- Better timing and positioning
- Scores improve steadily
- Q-values become more accurate

### Phase 4: Mastery (Episodes 5000+)
- High scores (50+ pipes)
- Reliable performance
- Optimal policy emerges
- Minimal exploration (low epsilon)

## Key Design Decisions

### Why DQN over other algorithms?
- **Simplicity**: Straightforward implementation
- **Proven**: Well-established algorithm (Mnih et al., 2015)
- **Efficiency**: Works well with discrete action spaces
- **Stability**: Target network and experience replay provide stability

### Why low-dimensional state over pixels?
- **Speed**: Much faster training and inference
- **Simplicity**: No need for CNNs or frame stacking
- **Sufficiency**: Contains all necessary information
- **Interpretability**: Easy to understand what agent sees

### Why this reward structure?
- **Shaped rewards**: Provide learning signal at every step
- **Balanced**: Not too sparse (only at pipe passes) or too dense (every pixel)
- **Clear objectives**: Agent knows what to optimize for

## Performance Metrics

### Training Metrics
- **Episode reward**: Cumulative reward per episode
- **Episode score**: Number of pipes passed
- **Episode length**: Number of steps survived
- **Loss**: Q-value prediction error
- **Epsilon**: Exploration rate

### Evaluation Metrics
- **Mean score**: Average pipes passed over evaluation episodes
- **Max score**: Best single-episode performance
- **Consistency**: Standard deviation of scores
- **Percentiles**: Distribution of performance

## File Structure

```
Flappy-bird-python/
├── flappy.py                 # Original game (reference)
├── flappy_env.py           # RL environment wrapper
├── dqn_agent.py            # DQN agent implementation
├── train_dqn.py            # Standard training script
├── train_iterative.py      # Iterative training script
├── evaluate.py             # Evaluation and visualization
├── test_render.py          # Rendering test script
├── assets/                 # Game assets (sprites, audio)
│   ├── sprites/
│   └── audio/
├── models/                 # Trained model checkpoints
│   ├── dqn_best.pth
│   ├── dqn_final_best.pth
│   └── dqn_iteration_*.pth
└── requirements.txt        # Dependencies
```

## Dependencies

- **pygame**: Game rendering and sprite management
- **torch**: Deep learning framework
- **numpy**: Numerical computations
- **matplotlib**: Training curve visualization

## Usage Examples

### Training
```bash
# Iterative training (recommended)
python train_iterative.py

# Standard training
python train_dqn.py
```

### Evaluation
```bash
# Headless evaluation
python evaluate.py --model models/dqn_best.pth --episodes 50

# Watch agent play
python evaluate.py --model models/dqn_best.pth --watch --games 5
```

## Technical Challenges & Solutions

### Challenge 1: Agent Not Learning
**Problem**: Agent stuck at score 0, crashing immediately
**Solution**: 
- Improved reward shaping (larger pipe bonus, crash penalty)
- Faster epsilon decay (20k vs 50k steps)
- Lower learning rate (5e-4 vs 1e-3) for stability

### Challenge 2: Unstable Training
**Problem**: Q-values exploding, loss increasing
**Solution**:
- Gradient clipping (max norm 10)
- Target network updates (every 500 steps)
- Experience replay (breaks correlation)

### Challenge 3: Slow Learning
**Problem**: Takes too long to see improvement
**Solution**:
- Faster epsilon decay
- More frequent target updates
- Better reward structure

## Results

With the current configuration, the agent typically achieves:
- **Mean score**: 20-30 pipes (after ~2000 episodes)
- **Max score**: 50-100+ pipes (after ~5000 episodes)
- **Consistency**: Standard deviation decreases as agent improves

## Future Improvements

1. **Double DQN**: Reduces overestimation bias
2. **Dueling DQN**: Separates value and advantage estimation
3. **Prioritized Experience Replay**: Sample important transitions more often
4. **Noisy Networks**: Replace epsilon-greedy with learned exploration
5. **Distributional RL**: Model full return distribution

## References

- Mnih et al. (2015). "Human-level control through deep reinforcement learning." Nature.
- Van Hasselt et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.
- Schaul et al. (2016). "Prioritized Experience Replay." ICLR.

