"""
Training script for DQN agent on Flappy Bird environment.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import os

from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent


def train_dqn(
    num_episodes=2000,
    max_steps_per_ep=2000,
    target_update_freq=1000,
    batch_size=64,
    save_freq=500,
    device="cpu"
):
    """
    Train a DQN agent on Flappy Bird.
    
    Args:
        num_episodes: Number of training episodes
        max_steps_per_ep: Maximum steps per episode
        target_update_freq: Frequency (in steps) to update target network
        batch_size: Batch size for training
        save_freq: Frequency (in episodes) to save model
        device: Device to use (cpu or cuda)
    """
    # Initialize environment and agent
    env = FlappyBirdEnv(render=False)
    state = env.reset()
    state_dim = len(state)
    action_dim = 2
    
    agent = DQNAgent(state_dim, action_dim, device=device)
    
    # Training statistics
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    moving_avg_rewards = deque(maxlen=100)
    moving_avg_scores = deque(maxlen=100)
    
    global_step = 0
    
    print("Starting DQN training...")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("-" * 50)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        
        while not done and ep_steps < max_steps_per_ep:
            # Select action
            action = agent.act(state, training=True)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            agent.replay.push(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step(batch_size=batch_size)
            
            # Update target network periodically
            if global_step % target_update_freq == 0 and global_step > 0:
                agent.update_target()
            
            state = next_state
            ep_reward += reward
            ep_steps += 1
            global_step += 1
        
        # Record statistics
        episode_rewards.append(ep_reward)
        episode_scores.append(info.get('score', 0))
        episode_lengths.append(ep_steps)
        moving_avg_rewards.append(ep_reward)
        moving_avg_scores.append(info.get('score', 0))
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(moving_avg_rewards)
            avg_score = np.mean(moving_avg_scores)
            current_eps = agent.epsilon()
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {ep_reward:.1f} | "
                  f"Score: {info.get('score', 0)} | "
                  f"Avg Reward (100): {avg_reward:.1f} | "
                  f"Avg Score (100): {avg_score:.1f} | "
                  f"Epsilon: {current_eps:.3f} | "
                  f"Steps: {ep_steps}")
        
        # Save model periodically
        if (episode + 1) % save_freq == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/dqn_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    agent.save("models/dqn_final.pth")
    print("\nTraining completed! Final model saved.")
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_scores, episode_lengths)
    
    return agent, episode_rewards, episode_scores


def plot_training_curves(rewards, scores, lengths):
    """Plot training statistics."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Episode rewards
    axes[0].plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(rewards) >= 100:
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        axes[0].plot(range(99, len(rewards)), moving_avg, color='red', label='Moving Avg (100)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training: Episode Rewards')
    axes[0].legend()
    axes[0].grid(True)
    
    # Episode scores
    axes[1].plot(scores, alpha=0.3, color='green', label='Episode Score')
    if len(scores) >= 100:
        moving_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
        axes[1].plot(range(99, len(scores)), moving_avg, color='red', label='Moving Avg (100)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Training: Episode Scores (Pipes Passed)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Episode lengths
    axes[2].plot(lengths, alpha=0.3, color='purple', label='Episode Length')
    if len(lengths) >= 100:
        moving_avg = np.convolve(lengths, np.ones(100)/100, mode='valid')
        axes[2].plot(range(99, len(lengths)), moving_avg, color='red', label='Moving Avg (100)')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Steps')
    axes[2].set_title('Training: Episode Lengths')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("Training curves saved to 'training_curves.png'")
    plt.close()


if __name__ == "__main__":
    # Set device - prefer CUDA if available
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    print(f"Device: {device}")
    print("-" * 50)
    
    # Train the agent
    agent, rewards, scores = train_dqn(
        num_episodes=2000,
        max_steps_per_ep=2000,
        target_update_freq=1000,
        batch_size=64,
        save_freq=500,
        device=device
    )

