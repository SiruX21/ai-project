import pygame
import random
import numpy as np
from pygame.locals import *

# Game constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
SPEED = 20
GRAVITY = 2.5
GAME_SPEED = 15

GROUND_WIDTH = 2 * SCREEN_WIDTH
GROUND_HEIGHT = 100

PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150


class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        
        # Ensure pygame.display is initialized before loading images
        if not pygame.get_init():
            pygame.init()
        if pygame.display.get_surface() is None:
            # Create a minimal display if none exists
            try:
                pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
            except:
                pygame.display.set_mode((1, 1))
        
        self.images = [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()
        ]
        
        self.speed = SPEED
        self.current_image = 0
        self.image = pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        
        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH / 6
        self.rect[1] = SCREEN_HEIGHT / 2
    
    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY
        self.rect[1] += self.speed
    
    def bump(self):
        self.speed = -SPEED
    
    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]


class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)
        
        # Ensure pygame.display is initialized before loading images
        if not pygame.get_init():
            pygame.init()
        if pygame.display.get_surface() is None:
            try:
                pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
            except:
                pygame.display.set_mode((1, 1))
        
        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))
        
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        
        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize
        
        self.mask = pygame.mask.from_surface(self.image)
    
    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        
        # Ensure pygame.display is initialized before loading images
        if not pygame.get_init():
            pygame.init()
        if pygame.display.get_surface() is None:
            try:
                pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
            except:
                pygame.display.set_mode((1, 1))
        
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))
        
        self.mask = pygame.mask.from_surface(self.image)
        
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT
    
    def update(self):
        self.rect[0] -= GAME_SPEED


def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])


def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted


class FlappyBirdEnv:
    """
    Flappy Bird environment for reinforcement learning.
    
    State representation (5D):
    - bird_y: normalized bird y position (0-1)
    - bird_velocity: normalized bird velocity
    - distance_to_next_pipe_x: normalized horizontal distance to next pipe
    - gap_top_y: normalized top of gap y position
    - gap_bottom_y: normalized bottom of gap y position
    """
    
    def __init__(self, render=False):
        self.render_mode = render
        
        # Initialize pygame (required for sprites even without rendering)
        if not pygame.get_init():
            pygame.init()
        
        self.screen = None
        self.clock = None
        self.background = None
        
        if self.render_mode:
            # Create visible window for rendering
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Flappy Bird RL - AI Agent')
            self.background = pygame.image.load('assets/sprites/background-day.png')
            self.background = pygame.transform.scale(self.background, (SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            # Make sure window is visible (not minimized)
            pygame.display.flip()
        else:
            # Initialize in headless mode (no display)
            # Use a minimal surface for sprite operations (required for sprites to work)
            try:
                pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
            except:
                # Fallback if HIDDEN flag not available
                pygame.display.set_mode((1, 1))
                pygame.display.iconify()
        
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        self.bird_group = pygame.sprite.Group()
        self.bird = Bird()
        self.bird_group.add(self.bird)
        
        self.ground_group = pygame.sprite.Group()
        for i in range(2):
            ground = Ground(GROUND_WIDTH * i)
            self.ground_group.add(ground)
        
        self.pipe_group = pygame.sprite.Group()
        for i in range(2):
            pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])
        
        self.score = 0
        self.last_pipe_x = None
        self.done = False
        
        return self._get_state()
    
    def _get_state(self):
        """Extract state vector from current game state."""
        # Bird state
        bird_y = self.bird.rect[1] / SCREEN_HEIGHT  # Normalized to [0, 1]
        bird_velocity = self.bird.speed / 50.0  # Normalized (roughly)
        
        # Group pipes by x-position (pipes come in pairs)
        pipe_pairs = {}
        for pipe in self.pipe_group.sprites():
            x_pos = pipe.rect[0]
            if x_pos not in pipe_pairs:
                pipe_pairs[x_pos] = []
            pipe_pairs[x_pos].append(pipe)
        
        # Find the next pipe pair (closest pair ahead of bird)
        next_pipe_x = None
        min_dist = float('inf')
        
        for x_pos, pipes in pipe_pairs.items():
            # Check if any pipe in this pair is ahead of the bird
            if x_pos + PIPE_WIDTH > self.bird.rect[0]:
                dist = x_pos - self.bird.rect[0]
                if dist < min_dist:
                    min_dist = dist
                    next_pipe_x = x_pos
        
        if next_pipe_x is not None and next_pipe_x in pipe_pairs:
            pipes = pipe_pairs[next_pipe_x]
            # Find top and bottom pipes
            top_pipe = None
            bottom_pipe = None
            for pipe in pipes:
                if pipe.rect[1] < SCREEN_HEIGHT / 2:  # Top pipe
                    top_pipe = pipe
                else:  # Bottom pipe
                    bottom_pipe = pipe
            
            if top_pipe and bottom_pipe:
                distance_to_pipe = (next_pipe_x - self.bird.rect[0]) / SCREEN_WIDTH
                gap_top = (top_pipe.rect[1] + top_pipe.rect[3]) / SCREEN_HEIGHT
                gap_bottom = bottom_pipe.rect[1] / SCREEN_HEIGHT
            else:
                # Fallback if we can't find both pipes
                distance_to_pipe = 1.0
                gap_top = 0.3
                gap_bottom = 0.5
        else:
            # No pipe ahead, use default values
            distance_to_pipe = 1.0
            gap_top = 0.3
            gap_bottom = 0.5
        
        state = np.array([
            bird_y,
            bird_velocity,
            distance_to_pipe,
            gap_top,
            gap_bottom
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: 0 = no flap, 1 = flap
        
        Returns:
            next_state, reward, done, info
        """
        if self.done:
            return self._get_state(), 0, True, {}
        
        # Apply action
        if action == 1:
            self.bird.bump()
        
        # Update game objects
        self.bird_group.update()
        self.ground_group.update()
        self.pipe_group.update()
        
        # Check for pipe passing (score) - improved logic
        passed_pipe = False
        bird_x = self.bird.rect[0]
        
        # Group pipes by x-position to find pairs
        pipe_pairs = {}
        for pipe in self.pipe_group.sprites():
            x_pos = pipe.rect[0]
            if x_pos not in pipe_pairs:
                pipe_pairs[x_pos] = []
            pipe_pairs[x_pos].append(pipe)
        
        # Check if bird passed any pipe pair
        for x_pos, pipes in pipe_pairs.items():
            # Bird has passed this pipe pair if the pipe's right edge is behind the bird
            if x_pos + PIPE_WIDTH < bird_x:
                # Only count if we haven't already counted this pair
                if self.last_pipe_x is None or x_pos < self.last_pipe_x:
                    passed_pipe = True
                    self.score += 1
                    self.last_pipe_x = x_pos
                    break  # Only count one pair per step
        
        # Check for collisions
        crashed = False
        if (pygame.sprite.groupcollide(self.bird_group, self.ground_group, False, False, pygame.sprite.collide_mask) or
            pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False, False, pygame.sprite.collide_mask)):
            crashed = True
            self.done = True
        
        # Check if bird goes off screen (top or bottom)
        if self.bird.rect[1] < 0 or self.bird.rect[1] > SCREEN_HEIGHT:
            crashed = True
            self.done = True
        
        # Update ground and pipes (remove off-screen, add new)
        if is_off_screen(self.ground_group.sprites()[0]):
            self.ground_group.remove(self.ground_group.sprites()[0])
            new_ground = Ground(GROUND_WIDTH - 20)
            self.ground_group.add(new_ground)
        
        if is_off_screen(self.pipe_group.sprites()[0]):
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            pipes = get_random_pipes(SCREEN_WIDTH * 2)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])
            self.last_pipe_x = None  # Reset when new pipes are added
        
        # Compute reward - improved reward shaping
        if crashed:
            reward = -100.0  # Large penalty for crashing
        else:
            reward = 0.1  # Small living reward
            
            if passed_pipe:
                reward += 10.0  # Large bonus for passing a pipe
            
            # Reward for staying alive and getting closer to next pipe
            # Find closest pipe ahead
            closest_pipe_dist = float('inf')
            for pipe in self.pipe_group.sprites():
                if pipe.rect[0] + PIPE_WIDTH > self.bird.rect[0]:  # Pipe ahead
                    dist = pipe.rect[0] - self.bird.rect[0]
                    if dist < closest_pipe_dist:
                        closest_pipe_dist = dist
            
            # Small bonus for being alive (encourages survival)
            if closest_pipe_dist < float('inf'):
                # Bonus for getting closer to pipe (normalized)
                reward += 0.05 * (1.0 - min(closest_pipe_dist / SCREEN_WIDTH, 1.0))
        
        # Render if enabled
        if self.render_mode:
            self._render()
        
        next_state = self._get_state()
        info = {'score': self.score}
        
        return next_state, reward, self.done, info
    
    def _render(self):
        """Render the current game state."""
        if not self.render_mode or self.screen is None:
            return
        
        # Handle pygame events (to prevent window from freezing)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
        
        # Draw everything
        self.screen.blit(self.background, (0, 0))
        self.bird_group.draw(self.screen)
        self.pipe_group.draw(self.screen)
        self.ground_group.draw(self.screen)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(15)
    
    def close(self):
        """Close the environment and clean up."""
        if self.render_mode:
            pygame.quit()

