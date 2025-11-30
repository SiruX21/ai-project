"""
Iterative training script that trains, evaluates, and continues until
a near-perfect model is achieved.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import os
import time

from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent


def evaluate_agent(agent, env, episodes=20, device="cpu"):
    """
    Evaluate agent and return statistics.
    
    Returns:
        Dictionary with evaluation stats
    """
    scores = []
    rewards = []
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action = agent.act(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward
        
        scores.append(info.get('score', 0))
        rewards.append(ep_reward)
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'mean_reward': np.mean(rewards),
        'scores': scores,
        'rewards': rewards
    }


def train_iteration(agent, env, episodes_per_iteration=200, max_steps_per_ep=2000,
                   target_update_freq=1000, batch_size=64, device="cpu", show_progress=True):
    """
    Train the agent for one iteration.
    
    Returns:
        Training statistics for this iteration
    """
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    losses = []
    
    global_step = agent.total_steps
    
    for episode in range(episodes_per_iteration):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        
        while not done and ep_steps < max_steps_per_ep:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.replay.push(state, action, reward, next_state, done)
            loss = agent.train_step(batch_size=batch_size)
            
            if loss is not None:
                losses.append(loss)
            
            if agent.total_steps % target_update_freq == 0 and agent.total_steps > 0:
                agent.update_target()
                if show_progress:
                    print(f"    [Step {agent.total_steps}] Target network updated")
            
            state = next_state
            ep_reward += reward
            ep_steps += 1
        
        episode_rewards.append(ep_reward)
        episode_scores.append(info.get('score', 0))
        episode_lengths.append(ep_steps)
        
        # Show progress every 20 episodes
        if show_progress and (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_score = np.mean(episode_scores[-20:])
            max_score = np.max(episode_scores[-20:])
            buffer_size = len(agent.replay)
            buffer_capacity = agent.replay.buffer.maxlen if hasattr(agent.replay.buffer, 'maxlen') else 'N/A'
            print(f"    Episode {episode + 1}/{episodes_per_iteration} | "
                  f"Avg Reward: {avg_reward:.1f} | "
                  f"Avg Score: {avg_score:.2f} | "
                  f"Max Score: {max_score} | "
                  f"Replay Buffer: {buffer_size}/{buffer_capacity}")
    
    return {
        'rewards': episode_rewards,
        'scores': episode_scores,
        'lengths': episode_lengths,
        'mean_loss': np.mean(losses) if losses else 0.0,
        'episodes': episodes_per_iteration
    }


def train_until_perfect(
    target_mean_score=50,  # Target average score (pipes passed)
    target_max_score=100,  # Target max score to achieve at least once
    min_eval_episodes=20,  # Minimum episodes for evaluation
    episodes_per_iteration=200,  # Episodes to train before each evaluation
    max_iterations=50,  # Maximum number of training iterations
    max_steps_per_ep=2000,
    target_update_freq=1000,
    batch_size=64,
    device="cpu",
    save_freq=5,  # Save model every N iterations
    patience=3,  # Stop if no improvement for N consecutive iterations
    show_training_progress=True,  # Show progress during training
    show_demo=False,  # Show live demo after each iteration
    demo_after_iterations=1,  # Show demo every N iterations
):
    """
    Iteratively train and evaluate until near-perfect performance is achieved.
    
    Args:
        target_mean_score: Target average score to achieve
        target_max_score: Target maximum score to achieve at least once
        min_eval_episodes: Number of episodes for evaluation
        episodes_per_iteration: Episodes to train before each evaluation
        max_iterations: Maximum number of training iterations
        max_steps_per_ep: Maximum steps per episode
        target_update_freq: Frequency to update target network
        batch_size: Batch size for training
        device: Device to use (cpu or cuda)
        save_freq: Save model every N iterations
        patience: Stop if no improvement for N consecutive iterations
    
    Returns:
        Best agent and training history
    """
    # Initialize environment and agent
    env = FlappyBirdEnv(render=False)
    state = env.reset()
    state_dim = len(state)
    action_dim = 2
    
    # Use slightly lower learning rate for stability
    agent = DQNAgent(state_dim, action_dim, device=device, lr=5e-4)
    
    # Training history
    history = {
        'iterations': [],
        'eval_mean_scores': [],
        'eval_max_scores': [],
        'eval_min_scores': [],
        'eval_std_scores': [],
        'train_mean_rewards': [],
        'train_mean_scores': [],
        'mean_losses': [],
        'best_score': -np.inf,
        'best_iteration': 0
    }
    
    best_agent_state = None
    no_improvement_count = 0
    target_reached = False  # Track if target has been achieved
    
    print("=" * 70)
    print("ITERATIVE TRAINING: Training for MAXIMUM performance")
    print("=" * 70)
    print(f"Target: Mean Score >= {target_mean_score}, Max Score >= {target_max_score}")
    print(f"⚠️  PUSHING FOR HIGHEST POSSIBLE SCORE - NO LIMIT!")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Episodes per iteration: {episodes_per_iteration}")
    print(f"Max iterations: {max_iterations}")
    print(f"Max steps per episode: {max_steps_per_ep}")
    print("=" * 70)
    
    os.makedirs("models", exist_ok=True)
    
    for iteration in range(max_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print(f"{'='*70}")
        
        # Training phase
        print(f"\n[Training Phase]")
        print(f"Training for {episodes_per_iteration} episodes...")
        train_start = time.time()
        
        train_stats = train_iteration(
            agent, env, episodes_per_iteration, max_steps_per_ep,
            target_update_freq, batch_size, device, show_training_progress
        )
        
        train_time = time.time() - train_start
        print(f"Training completed in {train_time:.1f}s")
        print(f"  Mean reward: {np.mean(train_stats['rewards']):.2f}")
        print(f"  Mean score: {np.mean(train_stats['scores']):.2f}")
        print(f"  Max score: {np.max(train_stats['scores'])}")
        print(f"  Min score: {np.min(train_stats['scores'])}")
        print(f"  Mean loss: {train_stats['mean_loss']:.4f}")
        print(f"  Epsilon: {agent.epsilon():.4f}")
        buffer_size = len(agent.replay)
        buffer_capacity = agent.replay.buffer.maxlen if hasattr(agent.replay.buffer, 'maxlen') else 'N/A'
        print(f"  Replay buffer size: {buffer_size}/{buffer_capacity}")
        print(f"  Total training steps: {agent.total_steps}")
        
        # Evaluation phase
        print(f"\n[Evaluation Phase]")
        print(f"Evaluating over {min_eval_episodes} episodes...")
        eval_start = time.time()
        
        eval_stats = evaluate_agent(agent, env, min_eval_episodes, device)
        
        eval_time = time.time() - eval_start
        print(f"Evaluation completed in {eval_time:.1f}s")
        print(f"  Mean score: {eval_stats['mean_score']:.2f} ± {eval_stats['std_score']:.2f}")
        print(f"  Max score: {eval_stats['max_score']}")
        print(f"  Min score: {eval_stats['min_score']}")
        print(f"  Mean reward: {eval_stats['mean_reward']:.2f}")
        
        # Record history
        history['iterations'].append(iteration + 1)
        history['eval_mean_scores'].append(eval_stats['mean_score'])
        history['eval_max_scores'].append(eval_stats['max_score'])
        history['eval_min_scores'].append(eval_stats['min_score'])
        history['eval_std_scores'].append(eval_stats['std_score'])
        history['train_mean_rewards'].append(np.mean(train_stats['rewards']))
        history['train_mean_scores'].append(np.mean(train_stats['scores']))
        history['mean_losses'].append(train_stats['mean_loss'])
        
        # Check for improvement
        current_best = eval_stats['max_score']
        improved = False
        
        if current_best > history['best_score']:
            history['best_score'] = current_best
            history['best_iteration'] = iteration + 1
            best_agent_state = agent.q_net.state_dict().copy()
            no_improvement_count = 0
            improved = True
            print(f"\n✓ NEW BEST SCORE: {current_best} (Iteration {iteration + 1})")
            
            # Save best model
            agent.save("models/dqn_best.pth")
            print(f"  Best model saved to models/dqn_best.pth")
        else:
            no_improvement_count += 1
            print(f"\n  No improvement (patience: {no_improvement_count}/{patience})")
        
        # Save model periodically
        if (iteration + 1) % save_freq == 0:
            agent.save(f"models/dqn_iteration_{iteration + 1}.pth")
            print(f"  Checkpoint saved to models/dqn_iteration_{iteration + 1}.pth")
        
        # Show live demo if enabled
        if show_demo and (iteration + 1) % demo_after_iterations == 0:
            print(f"\n{'='*70}")
            print("LIVE DEMONSTRATION: Showing agent play")
            print(f"{'='*70}")
            try:
                # Close existing environment first to avoid conflicts
                env.close()
                demo_env = FlappyBirdEnv(render=True)
                demo_state = demo_env.reset()
                demo_done = False
                demo_score = 0
                demo_steps = 0
                
                print("Watch the pygame window to see the agent play!")
                print("(Close window or press ESC to skip demo)")
                time.sleep(1)  # Give window time to appear
                
                while not demo_done and demo_steps < 500:  # Limit demo length
                    demo_action = agent.act(demo_state, training=False)
                    demo_next_state, demo_reward, demo_done, demo_info = demo_env.step(demo_action)
                    demo_state = demo_next_state
                    demo_score = demo_info.get('score', 0)
                    demo_steps += 1
                    time.sleep(0.05)  # Small delay for visibility
                
                print(f"Demo finished! Score: {demo_score}, Steps: {demo_steps}")
                demo_env.close()
                # Recreate training environment
                env = FlappyBirdEnv(render=False)
            except Exception as e:
                print(f"Demo failed: {e}")
                print("Continuing with training...")
                # Recreate training environment
                env = FlappyBirdEnv(render=False)
            print(f"{'='*70}\n")
        
        # Check if target achieved
        target_achieved = (
            eval_stats['mean_score'] >= target_mean_score and
            eval_stats['max_score'] >= target_max_score
        )
        
        if target_achieved and not target_reached:
            print(f"\n{'='*70}")
            print("🎉 TARGET ACHIEVED! 🎉")
            print(f"{'='*70}")
            print(f"Mean Score: {eval_stats['mean_score']:.2f} >= {target_mean_score} ✓")
            print(f"Max Score: {eval_stats['max_score']} >= {target_max_score} ✓")
            print(f"Total iterations: {iteration + 1}")
            print(f"Total episodes: {(iteration + 1) * episodes_per_iteration}")
            print(f"\n⚠️  Target achieved! Continuing to push for MAXIMUM possible score...")
            print(f"    (Will continue until no improvement for {patience} iterations)")
            target_reached = True
            # Reset patience counter to allow more training after target
            no_improvement_count = 0
        
        # Check patience (early stopping if no improvement)
        # Only stop if we've done at least 10 iterations and no improvement
        if no_improvement_count >= patience and iteration >= 10:
            print(f"\n{'='*70}")
            print(f"⚠️  EARLY STOPPING: No improvement for {patience} consecutive iterations")
            print(f"{'='*70}")
            print(f"Best score achieved: {history['best_score']} (Iteration {history['best_iteration']})")
            print(f"Final mean score: {eval_stats['mean_score']:.2f}")
            break
    
    # Load best model
    if best_agent_state is not None:
        agent.q_net.load_state_dict(best_agent_state)
        agent.target_net.load_state_dict(best_agent_state)
        agent.save("models/dqn_final_best.pth")
        print(f"\nFinal best model loaded and saved to models/dqn_final_best.pth")
    
    # Final evaluation - more comprehensive
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")
    print("Running comprehensive evaluation over 100 episodes...")
    final_eval = evaluate_agent(agent, env, episodes=100, device=device)
    print(f"\nFinal Statistics (100 episodes):")
    print(f"  Mean Score: {final_eval['mean_score']:.2f} ± {final_eval['std_score']:.2f}")
    print(f"  Max Score: {final_eval['max_score']} 🏆")
    print(f"  Min Score: {final_eval['min_score']}")
    print(f"  Median Score: {np.median(final_eval['scores']):.2f}")
    print(f"  75th Percentile: {np.percentile(final_eval['scores'], 75):.2f}")
    print(f"  90th Percentile: {np.percentile(final_eval['scores'], 90):.2f}")
    print(f"  Episodes with score > 50: {sum(1 for s in final_eval['scores'] if s > 50)}")
    print(f"  Episodes with score > 100: {sum(1 for s in final_eval['scores'] if s > 100)}")
    print(f"  Episodes with score > 200: {sum(1 for s in final_eval['scores'] if s > 200)}")
    
    # Final live demonstration
    print(f"\n{'='*70}")
    print("FINAL LIVE DEMONSTRATION")
    print(f"{'='*70}")
    print("Showing final trained agent playing...")
    final_demo_env = FlappyBirdEnv(render=True)
    
    for demo_game in range(3):
        demo_state = final_demo_env.reset()
        demo_done = False
        demo_score = 0
        demo_steps = 0
        
        print(f"\nDemo Game {demo_game + 1}/3 - Watch the pygame window!")
        time.sleep(1)
        
        while not demo_done and demo_steps < 1000:
            demo_action = agent.act(demo_state, training=False)
            demo_next_state, demo_reward, demo_done, demo_info = final_demo_env.step(demo_action)
            demo_state = demo_next_state
            demo_score = demo_info.get('score', 0)
            demo_steps += 1
            time.sleep(0.05)
        
        print(f"Demo Game {demo_game + 1} finished! Score: {demo_score}, Steps: {demo_steps}")
        if demo_game < 2:
            time.sleep(2)
    
    final_demo_env.close()
    print(f"\n{'='*70}")
    
    # Plot training history
    plot_iterative_training(history)
    
    return agent, history


def plot_iterative_training(history):
    """Plot iterative training progress."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    iterations = history['iterations']
    
    # Evaluation scores
    axes[0, 0].plot(iterations, history['eval_mean_scores'], 'o-', label='Mean Score', linewidth=2)
    axes[0, 0].fill_between(iterations,
                           np.array(history['eval_mean_scores']) - np.array(history['eval_std_scores']),
                           np.array(history['eval_mean_scores']) + np.array(history['eval_std_scores']),
                           alpha=0.3)
    axes[0, 0].plot(iterations, history['eval_max_scores'], 's-', label='Max Score', linewidth=2)
    axes[0, 0].axhline(y=history['best_score'], color='r', linestyle='--', label=f"Best: {history['best_score']}")
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Evaluation Scores Over Iterations')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training vs Evaluation scores
    axes[0, 1].plot(iterations, history['train_mean_scores'], 'o-', label='Train Mean Score', alpha=0.7)
    axes[0, 1].plot(iterations, history['eval_mean_scores'], 's-', label='Eval Mean Score', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Training vs Evaluation Scores')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Training rewards
    axes[1, 0].plot(iterations, history['train_mean_rewards'], 'o-', color='green', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Training Mean Rewards')
    axes[1, 0].grid(True)
    
    # Loss
    axes[1, 1].plot(iterations, history['mean_losses'], 'o-', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Training Loss')
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('iterative_training_curves.png', dpi=150)
    print("\nTraining curves saved to 'iterative_training_curves.png'")
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
    print("-" * 70)
    
    # Train iteratively until near-perfect
    print("\nStarting iterative training...")
    print("This will train the agent until it achieves near-perfect performance.")
    print("You'll see progress updates every 20 episodes during training.\n")
    
    agent, history = train_until_perfect(
        target_mean_score=200,     # Very high target - push for maximum performance
        target_max_score=500,      # Extremely high target - no limit
        min_eval_episodes=30,      # More episodes for better evaluation
        episodes_per_iteration=200, # Train 200 episodes per iteration
        max_iterations=100,        # More iterations to keep improving
        max_steps_per_ep=5000,     # Allow longer episodes for high scores
        target_update_freq=500,   # Update target more frequently
        batch_size=64,
        device=device,
        save_freq=5,               # Save checkpoint every 5 iterations
        patience=15,               # More patience to allow for breakthroughs
        show_training_progress=True, # Show progress during training
        show_demo=True,            # Show live demo after each iteration
        demo_after_iterations=3,   # Show demo every 3 iterations (less frequent)
    )
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best score: {history['best_score']} (Iteration {history['best_iteration']})")
    print(f"Total iterations: {len(history['iterations'])}")
    print(f"Final model saved to: models/dqn_final_best.pth")

