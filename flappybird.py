import pygame
import random

# --- Global Pygame Setup (can be kept global as it's typically one window) ---
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Flappy Bird")
clock = pygame.time.Clock() # Make clock global or pass it around

highscore = 0

class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 34
        self.height = 24
        self.gravity = 0.4
        self.jump_strength = -7
        self.velocity_y = 0
        # self.dead = False # No longer needed here, controlled by env

    def jump(self):
        self.velocity_y = self.jump_strength

    def update(self):
        self.velocity_y += self.gravity
        self.y += self.velocity_y
        # Removed self.dead check here, handled by Game class or Env for death conditions.

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 0), (self.x, self.y, self.width, self.height))

class Pipe:
    def __init__(self, x, height):
        self.x = x
        self.height = height
        self.width = 52
        self.gap = 150
        self.scored = False

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 255, 0), (self.x, 0, self.width, self.height))
        pygame.draw.rect(screen, (0, 255, 0), (self.x, self.height + self.gap, self.width, screen.get_height() - self.height - self.gap))

    # Removed __del__ as it's not useful here.

    def update_and_check(self, bird):
        """
        Updates pipe position and checks for collision/scoring.
        Returns: is_colliding, just_scored, remove_pipe
        """
        is_colliding = False
        just_scored = False
        
        self.x -= 4 # Pipe moves

        # Collision detection
        if self.x < bird.x + bird.width and self.x + self.width > bird.x:
            if bird.y < self.height or bird.y + bird.height > self.height + self.gap:
                is_colliding = True 

        # Scoring check
        if self.x + self.width < bird.x and not self.scored:
            just_scored = True
            self.scored = True

        # Check if pipe is off-screen
        remove_pipe = (self.x < -self.width)
        
        return is_colliding, just_scored, remove_pipe

class Game:
    def __init__(self):
        self.bird = Bird(50, 300)
        self.pipes = []
        self.score = 0
        self.frame = 0 # To track frames for pipe spawning
        
        # Initial pipe setup
        self.pipes.append(Pipe(screen.get_width() + 100, random.randint(100, 400))) # Initial pipe

    def reset(self):
        """Resets the game to its initial state for a new episode."""
        self.bird = Bird(50, 300)
        self.pipes = []
        self.score = 0
        self.frame = 0
        self.pipes.append(Pipe(screen.get_width() + 100, random.randint(100, 400))) # Initial pipe
        
    def get_game_state_for_env(self):
        """
        Returns relevant game state information needed by the FlappyBirdEnv.
        This is a cleaner way to pass data than direct attribute access.
        """
        # Find the next upcoming pipe
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + pipe.width > self.bird.x: # Pipe is still relevant
                next_pipe = pipe
                break
        
        return {
            'bird_y': self.bird.y,
            'bird_velocity_y': self.bird.velocity_y,
            'next_pipe': {
                'x': next_pipe.x if next_pipe else screen.get_width(),
                'gap_y': (next_pipe.height + next_pipe.gap // 2) if next_pipe else screen.get_height() // 2,
                'width': next_pipe.width if next_pipe else 0 
            },
            'pipes_list': self.pipes # Pass the whole list for the Env to iterate if needed
        }


    def update_game_state_one_frame(self, action):
        """
        Updates the game state by one frame based on the action.
        Returns game_over status and reward.
        """
        global highscore
        game_over = False
        reward = 0.1 # Base reward for staying alive

        # Apply action
        if action == 1: # Jump
            self.bird.jump()

        self.bird.update()

        # Check for bird hitting ground/ceiling
        if self.bird.y > screen.get_height() - self.bird.height or self.bird.y < 0:
            game_over = True
            reward = -100 # Penalty for hitting ground/ceiling

        # Pipe generation
        self.frame += 1
        if self.frame % 100 == 0: # Spawn a new pipe every 100 frames
            self.pipes.append(Pipe(screen.get_width(), random.randint(100, 400)))

        pipes_to_remove = []
        for pipe in self.pipes:
            colliding, scored, remove = pipe.update_and_check(self.bird)
            if colliding:
                game_over = True
                reward = -100 # Penalty for hitting pipe
            if scored:
                self.score += 1
                if self.score > highscore:
                    highscore = self.score
                reward = 10 # Reward for passing pipe
            if remove:
                pipes_to_remove.append(pipe)
        
        # Remove old pipes
        for pipe_to_del in pipes_to_remove:
            self.pipes.remove(pipe_to_del)

        return game_over, reward

    def draw(self):
        """Draws all game elements."""
        screen.fill((135, 206, 235))  # Sky color
        self.bird.draw(screen)
        for pipe in self.pipes:
            pipe.draw(screen)
        
        font = pygame.font.Font(None, 30) # Using default font
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        highscore_text = font.render(f"Highscore: {highscore}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        screen.blit(highscore_text, (10, 40))

        pygame.display.flip()
        clock.tick(60)

# Removed the global `game = Game()` and `reset_game()` from here.
# The AI environment will now instantiate and manage the Game object.