import pygame
import random
import sys


# ==========================================
# Configuration
# ==========================================

CELL_SIZE = 20

GRID_WIDTH = 20
GRID_HEIGHT = 20

WINDOW_WIDTH = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT

FPS = 8


# ==========================================
# Colors
# ==========================================

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

GREEN = (0, 200, 0)
DARK_GREEN = (0, 150, 0)

RED = (220, 0, 0)

GRAY = (40, 40, 40)


# ==========================================
# Snake Game
# ==========================================

class SnakeGame:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )

        pygame.display.set_caption("Snake RL - Basic Game")

        self.clock = pygame.time.Clock()

        self.font = pygame.font.Font(None, 28)

        self.reset()


    # ==========================================
    # Reset
    # ==========================================

    def reset(self):

        center_x = GRID_WIDTH // 2
        center_y = GRID_HEIGHT // 2

        self.snake = [
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y)
        ]

        self.direction = (1, 0)

        self.score = 0

        self.game_over = False

        self.spawn_food()

        return self.get_state()


    # ==========================================
    # Spawn food
    # ==========================================

    def spawn_food(self):

        while True:

            food = (
                random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1)
            )

            if food not in self.snake:

                self.food = food

                break


    # ==========================================
    # Change direction
    # ==========================================

    def change_direction(self, new_direction):

        current_x, current_y = self.direction

        new_x, new_y = new_direction

        # Prevent immediate 180-degree turn

        if (
            current_x + new_x == 0
            and
            current_y + new_y == 0
        ):
            return

        self.direction = new_direction

    
    def get_state(self):
        head_x,head_y=self.snake[0]

        direction_x,direction_y=self.direction

        # Directions relative to the current direction
        left_direction=(direction_y,-direction_x)
        right_direction=(-direction_y,direction_x)

        # Positions directly in front, to the left,and to the right
        straight_position=(
            head_x+direction_x,
            head_y+direction_y
        )

        left_position=(
            head_x+left_direction[0],
            head_y+left_direction[1]
        )

        right_position=(
            head_x+right_direction[0],
            head_y+right_direction[1]
        )

        # check whether a position is dangerous
        def is_danger(position):
            x,y=position

            # Wall collision
            if x<0 or x>=GRID_WIDTH or y<0 or y>=GRID_HEIGHT:
                return 1

            # Snake body collision
            if position in self.snake[1:]:
                return 1

            return 0
        
        danger_straight=is_danger(straight_position)
        danger_left=is_danger(left_position)
        danger_right=is_danger(right_position)

        #Current direction
        moving_left=direction_x==-1
        moving_right=direction_x==1
        moving_up=direction_y==-1
        moving_down=direction_y==1

        # Food position relative to the head
        food_left=self.food[0]<head_x
        food_right=self.food[0]>head_x
        food_up=self.food[1]<head_y
        food_down=self.food[1]>head_y

        state=[
            danger_straight,
            danger_left,
            danger_right,

            moving_left,
            moving_right,
            moving_up,
            moving_down,

            food_left,
            food_right,
            food_up,
            food_down
        ]

        return state

    def step(self, action):
        # Convert action into direction

        direction_x, direction_y = self.direction

        if action == 0:
            # Turn left
            self.direction = (direction_y, -direction_x)

        elif action == 1:
            # Go straight
            pass

        elif action == 2:
            # Turn right
            self.direction = (-direction_y, direction_x)

        else:
            raise ValueError("Invalid action")

        # Move the snake
        ate_food = self.move()

        # Check collision
        if self.check_collision():
            reward = -10
            done = True
            next_state = self.get_state()

            return next_state, reward, done

        # Reward for eating food
        if ate_food:
            reward = 10
        else:
            reward = 0

        done = False

        # Get new state
        next_state = self.get_state()

        return next_state, reward, done

    # ==========================================
    # Move
    # ==========================================

    def move(self):
        head_x, head_y = self.snake[0]
        
        direction_x, direction_y = self.direction

        new_head = (
            head_x + direction_x,
            head_y + direction_y
        )

        self.snake.insert(0, new_head)

        ate_food = new_head == self.food

        if ate_food:
            self.score += 1
            self.spawn_food()
        else:
            self.snake.pop()

        return ate_food

    # ==========================================
    # Collision
    # ==========================================

    def check_collision(self):

        head_x, head_y = self.snake[0]

        # Wall collision

        if (
            head_x < 0
            or head_x >= GRID_WIDTH
            or head_y < 0
            or head_y >= GRID_HEIGHT
        ):

            return True


        # Self collision

        if self.snake[0] in self.snake[1:]:

            return True


        return False


    # ==========================================
    # Keyboard input
    # ==========================================

    def handle_events(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                pygame.quit()

                sys.exit()


            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_UP:

                    self.change_direction((0, -1))


                elif event.key == pygame.K_DOWN:

                    self.change_direction((0, 1))


                elif event.key == pygame.K_LEFT:

                    self.change_direction((-1, 0))


                elif event.key == pygame.K_RIGHT:

                    self.change_direction((1, 0))


                elif event.key == pygame.K_r:

                    if self.game_over:

                        self.reset()


    # ==========================================
    # Draw grid
    # ==========================================

    def draw_grid(self):

        for x in range(
            0,
            WINDOW_WIDTH,
            CELL_SIZE
        ):

            pygame.draw.line(
                self.screen,
                GRAY,
                (x, 0),
                (x, WINDOW_HEIGHT)
            )


        for y in range(
            0,
            WINDOW_HEIGHT,
            CELL_SIZE
        ):

            pygame.draw.line(
                self.screen,
                GRAY,
                (0, y),
                (WINDOW_WIDTH, y)
            )


    # ==========================================
    # Draw Snake
    # ==========================================

    def draw_snake(self):

        for index, segment in enumerate(self.snake):

            x, y = segment

            rectangle = pygame.Rect(
                x * CELL_SIZE,
                y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )

            if index == 0:

                pygame.draw.rect(
                    self.screen,
                    DARK_GREEN,
                    rectangle
                )

            else:

                pygame.draw.rect(
                    self.screen,
                    GREEN,
                    rectangle
                )


    # ==========================================
    # Draw food
    # ==========================================

    def draw_food(self):

        x, y = self.food

        rectangle = pygame.Rect(
            x * CELL_SIZE,
            y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )

        pygame.draw.rect(
            self.screen,
            RED,
            rectangle
        )


    # ==========================================
    # Draw score
    # ==========================================

    def draw_score(self):

        text = self.font.render(
            f"Score: {self.score}",
            True,
            WHITE
        )

        self.screen.blit(
            text,
            (10, 10)
        )


    # ==========================================
    # Draw Game Over
    # ==========================================

    def draw_game_over(self):

        text = self.font.render(
            "GAME OVER - Press R to restart",
            True,
            WHITE
        )

        rectangle = text.get_rect(
            center=(
                WINDOW_WIDTH // 2,
                WINDOW_HEIGHT // 2
            )
        )

        self.screen.blit(
            text,
            rectangle
        )


    # ==========================================
    # Draw everything
    # ==========================================

    def draw(self):

        # Clear screen

        self.screen.fill(BLACK)

        # Draw game

        self.draw_grid()

        self.draw_snake()

        self.draw_food()

        self.draw_score()


        # Draw game-over message

        if self.game_over:

            self.draw_game_over()


        pygame.display.flip()


    # ==========================================
    # Main loop
    # ==========================================

    def run(self):

        while True:

            # 1. Read keyboard/events

            self.handle_events()


            # 2. Update game

            if not self.game_over:

                self.move()

                if self.check_collision():

                    self.game_over = True


            # 3. Render

            self.draw()


            # 4. Control speed

            self.clock.tick(FPS)


# ==========================================
# Start
# ==========================================

if __name__ == "__main__":

    game = SnakeGame()
    state=game.reset()
    print("Initial state",state)

    while True:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_r and game.game_over:
                    game.reset()
        if not game.game_over:
            action=random.randint(0,2)
            next_state,reward,done=game.step(action)
            
            if done:
                game.game_over=True
        game.draw()

        game.clock.tick(FPS)
            
            