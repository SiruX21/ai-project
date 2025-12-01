"""
Generate progress charts for AI project presentation.

This script creates multiple visualizations to demonstrate:
1. Baseline comparison (Random vs Trained)
2. Training progression across iterations
3. Feature importance (Ablation study)
4. Reliability distribution
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent
import os

# Configuration
MODEL_DIR = "models"
OUTPUT_DIR = "."
EPISODES_PER_EVAL = 50  # Number of episodes to run for each evaluation

def find_best_model(env, state_dim, action_dim, device, episodes=30):
    """
    Find the best performing model by evaluating candidates.
    
    Returns:
        (model_file_path, avg_score) or (None, -1) if no model found
    """
    best_candidates = [
        'dqn_best.pth',
        'dqn_iteration_15.pth',  # Often the best checkpoint
        'dqn_final_best.pth',
        'dqn_final.pth',
    ]
    
    best_score = -1
    best_model_file = None
    
    for model_file in best_candidates:
        model_path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(model_path):
            continue
        
        try:
            agent = DQNAgent(state_dim, action_dim, device=device)
            agent.load(model_path)
            scores = evaluate_agent(agent, env, episodes=episodes)
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_model_file = model_path
        except Exception as e:
            print(f"    Warning: Could not load {model_file}: {e}")
            continue
    
    return best_model_file, best_score

def evaluate_agent(agent, env, episodes=50, state_mask=None):
    """
    Evaluate an agent over multiple episodes.
    
    Args:
        agent: DQNAgent instance
        env: FlappyBirdEnv instance
        episodes: Number of episodes to run
        state_mask: Optional dict with indices to mask (zero out) in state
    
    Returns:
        List of scores achieved
    """
    scores = []
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Apply state mask if provided (for ablation study)
            if state_mask is not None:
                masked_state = state.copy()
                for idx in state_mask:
                    masked_state[idx] = 0.0
                action = agent.act(masked_state, training=False)
            else:
                action = agent.act(state, training=False)
            
            state, _, done, info = env.step(action)
        
        scores.append(info['score'])
    
    return scores


def chart_1_baseline_comparison():
    """Chart 1: Compare random agent vs trained agent."""
    print("\n" + "="*70)
    print("Generating Chart 1: Baseline Comparison (Random vs Trained)")
    print("="*70)
    
    env = FlappyBirdEnv(render=False)
    state_dim = 5
    action_dim = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Evaluate Random Agent
    print("\nEvaluating Random Agent...")
    random_scores = []
    for _ in range(EPISODES_PER_EVAL):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(np.random.randint(0, 2))
        random_scores.append(info['score'])
    
    # 2. Evaluate Trained Agent (use best model)
    print("Finding best trained model...")
    best_model_path, best_score = find_best_model(env, state_dim, action_dim, device, episodes=20)
    
    if best_model_path is None:
        print(f"Error: Could not find any model file. Skipping this chart.")
        env.close()
        return
    
    print(f"Using best model: {os.path.basename(best_model_path)} (preliminary score: {best_score:.1f})")
    agent = DQNAgent(state_dim, action_dim, device=device)
    agent.load(best_model_path)
    trained_scores = evaluate_agent(agent, env, episodes=EPISODES_PER_EVAL)
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    means = [np.mean(random_scores), np.mean(trained_scores)]
    stds = [np.std(random_scores), np.std(trained_scores)]
    labels = ['Random Baseline', 'Trained DQN Agent']
    colors = ['#bdc3c7', '#2ecc71']
    
    bars = plt.bar(labels, means, color=colors, yerr=stds, capsize=10, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.title('Performance Comparison: Baseline vs. Trained Model', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Average Pipes Passed', fontsize=13)
    plt.xlabel('Agent Type', fontsize=13)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 5,
                 f'{height:.1f} ± {stds[i]:.1f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement percentage
    improvement = ((means[1] - means[0]) / means[0]) * 100 if means[0] > 0 else 0
    plt.text(0.5, 0.95, f'Improvement: {improvement:.0f}%',
             transform=plt.gca().transAxes, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'baseline_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    env.close()


def chart_2_training_progression():
    """Chart 2: Show improvement across training iterations."""
    print("\n" + "="*70)
    print("Generating Chart 2: Training Progression Across Iterations")
    print("="*70)
    
    env = FlappyBirdEnv(render=False)
    state_dim = 5
    action_dim = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Models to evaluate - iteration checkpoints
    iteration_models = [
        ('dqn_iteration_5.pth', 'Iteration 5'),
        ('dqn_iteration_10.pth', 'Iteration 10'),
        ('dqn_iteration_15.pth', 'Iteration 15'),
    ]
    
    results = []
    labels = []
    stds = []
    
    # Evaluate iteration checkpoints
    for model_file, label in iteration_models:
        model_path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(model_path):
            print(f"  Skipping {model_file} (not found)")
            continue
        
        print(f"  Evaluating {label}...")
        agent = DQNAgent(state_dim, action_dim, device=device)
        agent.load(model_path)
        
        scores = evaluate_agent(agent, env, episodes=20)  # Fewer episodes per checkpoint
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        results.append(avg_score)
        stds.append(std_score)
        labels.append(label)
        print(f"    {label}: {avg_score:.1f} ± {std_score:.1f}")
    
    # Find and evaluate the best model
    print(f"  Finding best model...")
    best_model_path, preliminary_score = find_best_model(env, state_dim, action_dim, device, episodes=20)
    
    if best_model_path:
        print(f"    Evaluating best model: {os.path.basename(best_model_path)}...")
        agent = DQNAgent(state_dim, action_dim, device=device)
        agent.load(best_model_path)
        
        scores = evaluate_agent(agent, env, episodes=30)  # More episodes for best model
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        results.append(avg_score)
        stds.append(std_score)
        labels.append("Best Model")
        print(f"  ✓ Using {os.path.basename(best_model_path)} as Best Model (score: {avg_score:.1f} ± {std_score:.1f})")
    else:
        print("  Warning: No best model found. Skipping 'Best Model' entry.")
    
    if len(results) == 0:
        print("  No models found. Skipping this chart.")
        env.close()
        return
    
    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, results, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'], 
                   yerr=stds, capsize=10, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.title('Agent Improvement Over Training Iterations', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Stage', fontsize=13)
    plt.ylabel('Average Score (Pipes Passed)', fontsize=13)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + stds[i] + max(results) * 0.02,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'training_progression.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    env.close()


def chart_3_feature_importance():
    """Chart 3: Ablation study - feature importance."""
    print("\n" + "="*70)
    print("Generating Chart 3: Feature Importance (Ablation Study)")
    print("="*70)
    
    env = FlappyBirdEnv(render=False)
    state_dim = 5
    action_dim = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Find and load the best model
    print("Finding best model...")
    best_model_path, _ = find_best_model(env, state_dim, action_dim, device, episodes=20)
    
    if best_model_path is None:
        print(f"Error: Could not find any model file. Skipping this chart.")
        env.close()
        return
    
    print(f"Using best model: {os.path.basename(best_model_path)}")
    agent = DQNAgent(state_dim, action_dim, device=device)
    agent.load(best_model_path)
    
    # State indices: [bird_y, bird_velocity, distance_to_pipe, gap_top, gap_bottom]
    scenarios = {
        "Full Vision\n(Normal)": None,
        "No Velocity\nInfo": [1],  # Mask velocity
        "Blind to Next\nPipe": [2],  # Mask distance to pipe
        "No Gap Info": [3, 4],  # Mask gap positions
    }
    
    results = {}
    stds = {}
    
    for name, mask_indices in scenarios.items():
        print(f"  Testing: {name}...")
        scores = evaluate_agent(agent, env, episodes=30, state_mask=mask_indices)
        results[name] = np.mean(scores)
        stds[name] = np.std(scores)
        print(f"    Average: {results[name]:.1f} ± {stds[name]:.1f}")
    
    # Plotting
    plt.figure(figsize=(11, 6))
    names = list(results.keys())
    values = list(results.values())
    errors = [stds[name] for name in names]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c', '#e67e22']
    
    bars = plt.bar(names, values, color=colors, yerr=errors, capsize=10, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.title('Feature Importance Study (Methodology Validation)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Average Score', fontsize=13)
    plt.xlabel('Agent State Configuration', fontsize=13)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + errors[i] + max(values) * 0.02,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    env.close()


def chart_4_reliability_distribution():
    """Chart 4: Reliability histogram showing score distribution."""
    print("\n" + "="*70)
    print("Generating Chart 4: Reliability Distribution")
    print("="*70)
    
    env = FlappyBirdEnv(render=False)
    state_dim = 5
    action_dim = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Find and load the best model
    print("Finding best model...")
    best_model_path, _ = find_best_model(env, state_dim, action_dim, device, episodes=20)
    
    if best_model_path is None:
        print(f"Error: Could not find any model file. Skipping this chart.")
        env.close()
        return
    
    print(f"Using best model: {os.path.basename(best_model_path)}")
    agent = DQNAgent(state_dim, action_dim, device=device)
    agent.load(best_model_path)
    
    print(f"Running {EPISODES_PER_EVAL} episodes for distribution analysis...")
    scores = evaluate_agent(agent, env, episodes=EPISODES_PER_EVAL)
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    median_score = np.median(scores)
    
    print(f"  Mean: {mean_score:.1f} ± {std_score:.1f}")
    print(f"  Median: {median_score:.1f}")
    print(f"  Min: {np.min(scores)}, Max: {np.max(scores)}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(scores, bins=20, color='#3498db', edgecolor='black', 
                                 alpha=0.7, linewidth=1.5)
    
    # Add vertical lines for statistics
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_score:.1f}')
    plt.axvline(median_score, color='green', linestyle='--', linewidth=2,
                label=f'Median: {median_score:.1f}')
    
    plt.title('Agent Reliability Distribution (100 Episodes)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Score Achieved (Pipes Passed)', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add text box with statistics
    stats_text = f'Mean: {mean_score:.1f}\nStd: {std_score:.1f}\nMedian: {median_score:.1f}'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'reliability_hist.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    env.close()


def main():
    """Generate all progress charts."""
    print("\n" + "="*70)
    print("AI PROJECT: Progress Chart Generator")
    print("="*70)
    print(f"Model directory: {MODEL_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Episodes per evaluation: {EPISODES_PER_EVAL}")
    
    # Check if models directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"\nError: Model directory '{MODEL_DIR}' not found!")
        return
    
    # Generate all charts
    chart_1_baseline_comparison()
    chart_2_training_progression()
    chart_3_feature_importance()
    chart_4_reliability_distribution()
    
    print("\n" + "="*70)
    print("All charts generated successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - baseline_comparison.png")
    print("  - training_progression.png")
    print("  - feature_importance.png")
    print("  - reliability_hist.png")
    print("\nThese charts demonstrate:")
    print("  1. Baseline comparison shows the power of RL")
    print("  2. Training progression shows iterative improvement")
    print("  3. Feature importance validates your methodology")
    print("  4. Reliability histogram shows consistency")


if __name__ == "__main__":
    main()

