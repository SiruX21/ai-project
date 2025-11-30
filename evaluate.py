"""
Evaluation script for trained DQN agent.
Can run in headless mode (no rendering) or with rendering enabled.
"""
import numpy as np
import torch
import time
import argparse

from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent


def evaluate(agent, env, episodes=20, render=False, delay=0.05):
    """
    Evaluate a trained agent.
    
    Args:
        agent: Trained DQNAgent
        env: FlappyBirdEnv instance
        episodes: Number of evaluation episodes
        render: Whether to render the game
        delay: Delay between frames when rendering (seconds)
    
    Returns:
        Dictionary with evaluation statistics
    """
    scores = []
    rewards = []
    lengths = []
    
    print(f"Evaluating agent over {episodes} episodes...")
    print("-" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        
        while not done:
            # Use greedy policy (no exploration)
            action = agent.act(state, training=False)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            ep_reward += reward
            ep_steps += 1
            
            if render:
                time.sleep(delay)
        
        score = info.get('score', 0)
        scores.append(score)
        rewards.append(ep_reward)
        lengths.append(ep_steps)
        
        print(f"Episode {episode + 1}/{episodes} | "
              f"Score: {score} | "
              f"Reward: {ep_reward:.1f} | "
              f"Length: {ep_steps}")
    
    # Compute statistics
    stats = {
        'scores': scores,
        'rewards': rewards,
        'lengths': lengths,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
    }
    
    print("-" * 50)
    print("Evaluation Results:")
    print(f"  Mean Score: {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
    print(f"  Max Score: {stats['max_score']}")
    print(f"  Min Score: {stats['min_score']}")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean Episode Length: {stats['mean_length']:.2f} steps")
    
    return stats


def watch_agent_play(agent, env, num_games=3, delay=0.05):
    """
    Watch the agent play with rendering enabled (live demonstration).
    
    Args:
        agent: Trained DQNAgent
        env: FlappyBirdEnv instance (should have render=True)
        num_games: Number of games to watch
        delay: Delay between frames (seconds)
    """
    if not env.render_mode:
        print("WARNING: Environment is not in render mode! Window will not appear.")
        print("Reinitializing with render=True...")
        env.close()
        env = FlappyBirdEnv(render=True)
    
    print(f"\n{'='*70}")
    print("LIVE DEMONSTRATION: Watching agent play")
    print(f"{'='*70}")
    print(f"Watching agent play {num_games} games...")
    print("Close the window or press ESC to stop early.")
    print("The pygame window should be visible now!")
    print("-" * 70)
    
    # Small delay to ensure window appears
    time.sleep(0.5)
    
    for game in range(num_games):
        state = env.reset()
        done = False
        score = 0
        steps = 0
        
        print(f"\nGame {game + 1}/{num_games} - Starting...")
        print("Watch the pygame window to see the agent play!")
        
        while not done:
            action = agent.act(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            score = info.get('score', 0)
            steps += 1
            
            # Print progress every 50 steps
            if steps % 50 == 0:
                print(f"  Step {steps}: Score={score}, Reward={reward:.2f}")
            
            time.sleep(delay)
        
        print(f"Game {game + 1} finished! Final Score: {score}, Steps: {steps}")
        if game < num_games - 1:  # Don't pause after last game
            print("Starting next game in 2 seconds...")
            time.sleep(2)  # Pause between games
    
    print(f"\n{'='*70}")
    print("Demonstration complete!")
    print(f"{'='*70}")
    return env


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent')
    parser.add_argument('--model', type=str, default='models/dqn_final.pth',
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render the game during evaluation')
    parser.add_argument('--watch', action='store_true',
                        help='Watch the agent play with live demonstration (renders automatically)')
    parser.add_argument('--games', type=int, default=3,
                        help='Number of games to watch (only for --watch mode)')
    parser.add_argument('--demo', action='store_true',
                        help='Show live demonstration (same as --watch)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda). Auto-detects CUDA if not specified.')
    
    args = parser.parse_args()
    
    # Initialize environment
    render_mode = args.render or args.watch or args.demo
    print(f"Initializing environment with render={render_mode}...")
    env = FlappyBirdEnv(render=render_mode)
    
    if render_mode:
        print("✓ Render mode enabled - pygame window should be visible")
        print("  (If window doesn't appear, check if it's minimized or behind other windows)")
    
    state = env.reset()
    state_dim = len(state)
    action_dim = 2
    
    # Initialize agent and load weights - auto-detect CUDA by default
    if args.device is None:
        # Auto-detect: prefer CUDA if available
        if torch.cuda.is_available():
            device = "cuda"
            print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("CUDA not available, using CPU")
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
    
    agent = DQNAgent(state_dim, action_dim, device=device)
    
    try:
        agent.load(args.model)
        print(f"Loaded model from {args.model}")
    except FileNotFoundError:
        print(f"Error: Model file {args.model} not found!")
        print("Please train a model first using train_dqn.py")
        return
    
    # Run evaluation or watch mode
    if args.watch or args.demo:
        env = watch_agent_play(agent, env, num_games=args.games)
    else:
        stats = evaluate(agent, env, episodes=args.episodes, render=args.render)
    
    # Keep window open briefly before closing
    if args.render or args.watch:
        print("\nClosing window in 2 seconds...")
        time.sleep(2)
    
    env.close()


if __name__ == "__main__":
    main()

