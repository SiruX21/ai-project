# Chart Generation Guide

This guide explains how to generate progress charts for your AI project presentation.

## Quick Start

Run the chart generation script:

```bash
python generate_progress_charts.py
```

This will generate 4 charts in the current directory:
1. `baseline_comparison.png` - Random agent vs Trained agent comparison
2. `training_progression.png` - Improvement across training iterations
3. `feature_importance.png` - Ablation study showing which features matter
4. `reliability_hist.png` - Score distribution showing consistency

## Requirements

- Trained model files in the `models/` directory:
  - `dqn_final_best.pth` (or `dqn_final.pth`) - Required for most charts
  - `dqn_iteration_5.pth` - Optional, for training progression
  - `dqn_iteration_10.pth` - Optional, for training progression
  - `dqn_iteration_15.pth` - Optional, for training progression

## Chart Descriptions

### 1. Baseline Comparison (`baseline_comparison.png`)

**Purpose:** Shows the dramatic improvement from random actions to trained agent.

**What it shows:**
- Average score of random agent (baseline)
- Average score of trained DQN agent
- Improvement percentage

**Presentation narrative:**
- "This chart demonstrates the power of reinforcement learning. Our trained agent achieves scores that are X% better than random chance."

**Best for:** Results section, showing why RL works

---

### 2. Training Progression (`training_progression.png`)

**Purpose:** Shows iterative improvement during training.

**What it shows:**
- Average scores at different training checkpoints (Iteration 5, 10, 15, Final)
- Clear upward trend demonstrating learning

**Presentation narrative:**
- "At iteration 5, the agent struggled with an average score of X. By iteration 15, it improved to Y. The final model achieves Z, showing clear learning progression."

**Best for:** Results section, demonstrating progress

---

### 3. Feature Importance (`feature_importance.png`)

**Purpose:** Validates your methodology by showing which state features are critical.

**What it shows:**
- Full model performance (normal)
- Performance when velocity is masked
- Performance when pipe distance is masked
- Performance when gap information is masked

**Presentation narrative:**
- "This ablation study proves our state representation was well-designed. When we remove velocity information, performance drops significantly, proving that velocity is a crucial feature for decision-making."

**Best for:** Methodology section, validating design choices

---

### 4. Reliability Distribution (`reliability_hist.png`)

**Purpose:** Shows consistency and reliability of the trained agent.

**What it shows:**
- Histogram of scores over 100 episodes
- Mean, median, and standard deviation
- Distribution shape (shows if agent is consistent)

**Presentation narrative:**
- "The agent is not only high-performing but also consistent. Over 100 episodes, it maintains an average score of X with low variance, demonstrating stable learned behavior."

**Best for:** Results section, showing reliability

---

## Customization

You can modify the script to adjust:

- **Number of episodes per evaluation:** Change `EPISODES_PER_EVAL` (default: 50)
- **Model paths:** Modify `MODEL_DIR` if your models are elsewhere
- **Output directory:** Change `OUTPUT_DIR` to save charts elsewhere
- **Colors/styling:** Modify the matplotlib styling in each chart function

## Troubleshooting

### Error: "Model file not found"
- Ensure your model files are in the `models/` directory
- Check that filenames match exactly (case-sensitive)
- The script will try `dqn_final_best.pth` first, then `dqn_final.pth`

### Charts look different than expected
- Scores depend on your specific trained models
- If scores are very low, the agent may need more training
- If variance is high, consider running more evaluation episodes

### Script runs slowly
- Reduce `EPISODES_PER_EVAL` for faster generation
- Use GPU if available (script auto-detects CUDA)
- For training progression, fewer episodes per checkpoint is fine

## Integration with Presentation

### Slide Recommendations

1. **"The Power of RL" slide:**
   - Use `baseline_comparison.png`
   - Emphasize the improvement percentage

2. **"Training Progress" slide:**
   - Use `training_progression.png` or `iterative_training_curves.png`
   - Show clear learning trajectory

3. **"Why Our Methodology Works" slide:**
   - Use `feature_importance.png`
   - Explain that removing features proves their importance

4. **"Agent Reliability" slide:**
   - Use `reliability_hist.png`
   - Emphasize consistency and stability

## Tips

- Generate charts **before** creating slides to see what you have
- Use high-resolution charts (script saves at 300 DPI)
- Consider adding annotations or highlights in your presentation software
- Keep charts simple and readable - one main message per chart
- Practice explaining each chart - know what story it tells

## Example Usage

```bash
# Navigate to project directory
cd Flappy-bird-python

# Generate all charts
python generate_progress_charts.py

# Output:
# ======================================================================
# AI PROJECT: Progress Chart Generator
# ======================================================================
# Generating Chart 1: Baseline Comparison...
# ✓ Saved: baseline_comparison.png
# Generating Chart 2: Training Progression...
# ✓ Saved: training_progression.png
# Generating Chart 3: Feature Importance...
# ✓ Saved: feature_importance.png
# Generating Chart 4: Reliability Distribution...
# ✓ Saved: reliability_hist.png
# ======================================================================
# All charts generated successfully!
# ======================================================================
```

Good luck with your presentation! 🎯📊

