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

        # Add new head

        self.snake.insert(0, new_head)

        # Eat food

        if new_head == self.food:

            self.score += 1

            self.spawn_food()

        else:

            # Remove tail

            self.snake.pop()


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

    game.run()