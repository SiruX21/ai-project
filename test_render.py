"""
Simple test script to verify pygame rendering works.
"""
import time
from flappy_env import FlappyBirdEnv

print("Testing pygame rendering...")
print("=" * 50)

# Create environment with rendering enabled
print("Creating environment with render=True...")
env = FlappyBirdEnv(render=True)

print("Environment created. Window should be visible now!")
print("Running a few steps to test rendering...")
print("(Close the window or press ESC to stop)")

state = env.reset()
print(f"Initial state shape: {state.shape}")

for step in range(100):
    # Random action for testing
    import random
    action = random.randint(0, 1)
    
    next_state, reward, done, info = env.step(action)
    
    if step % 10 == 0:
        print(f"Step {step}: Score={info.get('score', 0)}, Reward={reward:.2f}, Done={done}")
    
    if done:
        print(f"Game over! Final score: {info.get('score', 0)}")
        state = env.reset()
        print("Reset - starting new game...")
    
    state = next_state
    time.sleep(0.05)  # Small delay to see the game

print("\nTest complete! Closing environment...")
env.close()
print("Done!")

