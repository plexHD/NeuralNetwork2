import random
import numpy as np # Added for potential use, and action_space_sample consistency
import math

class Maze: 
    def __init__(self, size): # size is (height, width)
        self.size = size
        self.structure = [] # 0 = empty space, 1 = wall, 2 = end point, 3 = start point
        
        # Define start position based on size, as in original
        self.start_pos_initial = (size[0] - 3, int(round(size[1] / 2, 0)))
        self.player_position = self.start_pos_initial  # Current player position
        self.memory = []
        self.distance_to_exit = 100
        
        self.generateMaze()

        # Gym-like attributes
        self.action_space_n = 4 # 0:W (left), 1:E (right), 2:N (up), 3:S (down)
        self.observation_space_n = self.size[0] * self.size[1] # state is player_y * width + player_x
        
        # For mapping agent actions (integers) to coordinate changes
        self._action_to_delta = {
            0: (0, -1), # West (dx = -1)
            1: (0, 1),  # East (dx = 1)
            2: (-1, 0), # North (dy = -1)
            3: (1, 0)   # South (dy = 1)
        }

    def generateMaze(self):
        self.structure = []
        for y_coord in range(self.size[0]):
            row = []
            # Obstacle count for this specific row, as in original file
            obstacle_count_for_row = random.randint(0, self.size[1] // 8)
            for x_coord in range(self.size[1]):
                is_start_point = (y_coord == self.start_pos_initial[0] and x_coord == self.start_pos_initial[1])
                
                if is_start_point:
                    row.append(3) # Start point
                elif y_coord == 0 or y_coord == self.size[0] - 1 or x_coord == 0 or x_coord == self.size[1] - 1:
                    row.append(1) # Border Wall
                # Original obstacle placement logic:
                elif not is_start_point and random.randint(0, self.size[1]) < obstacle_count_for_row :
                    row.append(1) # Wall
                else:
                    row.append(0) # Empty space
            self.structure.append(row)
        
        self.structure[self.start_pos_initial[0]][self.start_pos_initial[1]] = 3 # Ensure start point

        # Place end point, ensuring it's not the start point or a wall
        while True:
            random_row = random.randint(1, self.size[0] - 2) # Avoid borders
            random_col = random.randint(1, self.size[1] - 2) # Avoid borders
            if self.structure[random_row][random_col] == 0 and \
               (random_row, random_col) != self.start_pos_initial:
                self.structure[random_row][random_col] = 2  # End point
                self.end_pos = (random_row, random_col)
                break
        
        self.player_position = self.start_pos_initial # Reset player to start after generation

    def _get_state(self):
        # Make a deep copy of the structure to avoid modifying the original
        state = [row[:] for row in self.structure]
        state.append([self.player_position[0] * self.size[1] + self.player_position[1], self.distance_to_exit]) # Append player position as last element
        return state

    def reset(self):
        # self.generateMaze()
        self.memory = []
        self.player_position = self.start_pos_initial
        return self._get_state()

    def action_space_sample(self):
        return random.randint(0, self.action_space_n - 1)

    def step(self, action_idx):
        if not (0 <= action_idx < self.action_space_n):
            raise ValueError(f"Invalid action: {action_idx}")
        
        recent_memory = 5

        old_y, old_x = self.player_position
        dy, dx = self._action_to_delta[action_idx]
        new_y, new_x = old_y + dy, old_x + dx

        old_exit_distance = math.sqrt((self.end_pos[0] - old_y) ** 2 + (self.end_pos[1] - old_x) ** 2)
        new_exit_distance = math.sqrt((self.end_pos[0] - new_y) ** 2 + (self.end_pos[1] - new_x) ** 2)
        self.distance_to_exit = new_exit_distance

        reward = -0.1 # Small penalty for each step to encourage efficiency
        done = False

        if not (0 <= new_y < self.size[0] and 0 <= new_x < self.size[1]):
            reward = -0.2 
            # Player position does not change, state remains old state
            current_state = self._get_state() 
        else:
            cell_type = self.structure[new_y][new_x]
            if cell_type == 1: # Wall
                reward = -0.04
                # Player position does not change
            elif cell_type == 2: # End point
                reward = 4
                done = True
                self.player_position = (new_y, new_x) 
            elif cell_type == 0 or cell_type == 3: # Empty space or Start point
                # reward = -1
                if old_exit_distance > new_exit_distance:
                    reward += 0.1
                else:
                    reward -= 0.15
                if (new_y, new_x) in self.memory:
                    reward -= 0.25
                self.player_position = (new_y, new_x)

            current_state = self._get_state()
            self.memory.append((old_y, old_x))
            if len(self.memory) > recent_memory:
                self.memory.pop(0)

        info = {}
        return current_state, reward, done, info

    def render(self):
        temp_structure = [row[:] for row in self.structure]
        py, px = self.player_position
        if 0 <= py < self.size[0] and 0 <= px < self.size[1]:
            temp_structure[py][px] = "P" 
        
        for row in temp_structure:
            for cell_val in row:
                if cell_val == 0: print(" ", end="")
                elif cell_val == 1: print("█", end="")
                elif cell_val == 2: print("E", end="")
                elif cell_val == 3: print("S", end="")
                elif cell_val == "P": print("P", end="")
            print()
        
    def movePlayer(self, direction_char): # For manual play
        action_map_manual = {'w': 0, 'e': 1, 'n': 2, 's': 3}                                             
        
        if direction_char not in action_map_manual:
            print(f"Invalid direction \'{direction_char}\'! Use \'w\', \'e\', \'n\', or \'s\'.")
            return

        action_idx = action_map_manual[direction_char]
        
        old_y, old_x = self.player_position
        dy, dx = self._action_to_delta[action_idx]
        new_y, new_x = old_y + dy, old_x + dx

        if not (0 <= new_y < self.size[0] and 0 <= new_x < self.size[1]):
            print("Move out of bounds!")
            return

        cell_type = self.structure[new_y][new_x]
        if cell_type == 1:
            print("Collision with wall!")
            return 
        
        self.player_position = (new_y, new_x) 

        if cell_type == 2:
            print("Reached the end!")
        
        print(f"Player at: {self.player_position}")