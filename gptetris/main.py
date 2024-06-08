import pygame
import random
import json

# Load configuration from JSON file
with open('config.json') as config_file:
    config = json.load(config_file)

# Initialize Pygame
pygame.init()

# Screen dimensions and colors from config
SCREEN_WIDTH = config['screen']['width']
SCREEN_HEIGHT = config['screen']['height']
WHITE = tuple(config['colors']['white'])
BLACK = tuple(config['colors']['black'])
RED = tuple(config['colors']['red'])
GREEN = tuple(config['colors']['green'])
LIGHT_GRAY = tuple(config['colors']['light_gray'])

# Grid settings from config
GRID_WIDTH = config['grid']['width']
GRID_HEIGHT = config['grid']['height']
CELL_SIZE = config['grid']['cell_size']

# Shapes and colors from config
SHAPES = config['shapes']
SHAPE_COLORS = [tuple(color) for color in config['shape_colors']]

# Motivational phrases from config
PHRASES = config['phrases']

# File paths from config
background_music_path = config['file_paths']['background_music']
bg_language_path = config['file_paths']['bg_language']
bg_title_path = config['file_paths']['bg_title']
bg_instructions_path = config['file_paths']['bg_instructions']
bg_game_path = config['file_paths']['bg_game']
bg_game_over_path = config['file_paths']['bg_game_over']
bg_win_path = config['file_paths']['bg_win']

# Language selection
language = "en"

# Screen setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('GPTetris')

# Load and scale background images
bg_language = pygame.transform.scale(pygame.image.load(bg_language_path), (SCREEN_WIDTH, SCREEN_HEIGHT))
bg_title = pygame.transform.scale(pygame.image.load(bg_title_path), (SCREEN_WIDTH, SCREEN_HEIGHT))
bg_instructions = pygame.transform.scale(pygame.image.load(bg_instructions_path), (SCREEN_WIDTH, SCREEN_HEIGHT))
bg_game = pygame.transform.scale(pygame.image.load(bg_game_path), (SCREEN_WIDTH, SCREEN_HEIGHT))
bg_game_over = pygame.transform.scale(pygame.image.load(bg_game_over_path), (SCREEN_WIDTH, SCREEN_HEIGHT))
bg_win = pygame.transform.scale(pygame.image.load(bg_win_path), (SCREEN_WIDTH, SCREEN_HEIGHT))

# Load and play background music
pygame.mixer.music.load(background_music_path)
pygame.mixer.music.set_volume(0.1)  # Set volume to a lower level
pygame.mixer.music.play(-1)  # Play the music in a loop

# Define the Tetromino class
class Tetromino:
    def __init__(self, shape, color):
        self.shape = shape
        self.color = color
        self.x = GRID_WIDTH // 2 - len(shape[0]) // 2
        self.y = 0

    def draw(self, screen):
        for i, row in enumerate(self.shape):
            for j, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, self.color, pygame.Rect(
                        (self.x + j) * CELL_SIZE, (self.y + i) * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def rotate(self):
        self.shape = [list(row) for row in zip(*self.shape[::-1])]

# Define the game functions
def create_grid(locked_positions={}):
    grid = [[BLACK for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if (x, y) in locked_positions:
                grid[y][x] = locked_positions[(x, y)]
    return grid

def check_collision(tetromino, grid):
    for i, row in enumerate(tetromino.shape):
        for j, cell in enumerate(row):
            if cell:
                if tetromino.x + j < 0 or tetromino.x + j >= GRID_WIDTH or tetromino.y + i >= GRID_HEIGHT:
                    return True  # Collision with boundaries
                if grid[tetromino.y + i][tetromino.x + j] != BLACK:
                    return True  # Collision with locked positions
    return False

def clear_rows(grid, locked_positions):
    full_rows = [i for i, row in enumerate(grid) if BLACK not in row]
    if full_rows:
        for row in full_rows:
            for x in range(GRID_WIDTH):
                del locked_positions[(x, row)]

        # Shift down rows above the cleared rows
        for row in sorted(full_rows):
            for y in range(row, -1, -1):
                for x in range(GRID_WIDTH):
                    if (x, y) in locked_positions:
                        locked_positions[(x, y + 1)] = locked_positions.pop((x, y))
    return len(full_rows)

def render_text_with_border(screen, text, font, color, border_color, x, y):
    text_surface = font.render(text, True, color)
    text_surface_border = font.render(text, True, border_color)
    screen.blit(text_surface_border, (x + 2, y + 2))
    screen.blit(text_surface, (x, y))

# Function to display the game over screen
def game_over_screen(screen, score):
    screen.blit(bg_game_over, (0, 0))
    font = pygame.font.SysFont('Comic Sans MS', 48)
    game_over_text = "GAME OVER" if language == "en" else "FIM DE JOGO"
    score_text = f"Score: {score}" if language == "en" else f"Pontos: {score}"
    render_text_with_border(screen, game_over_text, font, RED, BLACK, SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 50)
    render_text_with_border(screen, score_text, font, RED, BLACK, SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 10)
    pygame.display.flip()
    pygame.time.wait(3000)
    main_menu(screen)

# Function to display the winning screen
def win_screen(screen):
    screen.blit(bg_win, (0, 0))
    font = pygame.font.SysFont('Comic Sans MS', 48)
    win_text = "YOU WIN!" if language == "en" else "VOCÊ VENCEU!"
    congrats_text = "Congratulations!" if language == "en" else "Parabéns!"
    render_text_with_border(screen, win_text, font, GREEN, BLACK, SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 50)
    render_text_with_border(screen, congrats_text, font, GREEN, BLACK, SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 10)
    pygame.display.flip()
    pygame.time.wait(3000)
    main_menu(screen)

# Function to display the pause screen
def pause_screen(screen):
    font = pygame.font.SysFont('Comic Sans MS', 48)
    pause_text = "PAUSED" if language == "en" else "PAUSADO"
    render_text_with_border(screen, pause_text, font, WHITE, BLACK, SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2)
    pygame.display.flip()

# Function to display the title screen
def title_screen(screen):
    screen.blit(bg_title, (0, 0))
    font = pygame.font.SysFont('Comic Sans MS', 24)
    render_text_with_border(screen, "Press any key to start", font, WHITE, BLACK, SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 50)
    pygame.display.flip()

# Function to display the instructions screen
def instructions_screen(screen):
    screen.blit(bg_instructions, (0, 0))
    font = pygame.font.SysFont('Comic Sans MS', 24)
    if language == "en":
        instructions = [
            "Use the arrow keys to move",
            "Up arrow to rotate",
            "Down arrow to move down",
            "Space to drop quickly",
            "Enter to pause",
            "ESC to quit"
        ]
        prompt_text = "Press ENTER to start"
    else:
        instructions = [
            "Use as setas para mover",
            "Seta para cima para girar",
            "Seta para baixo para descer",
            "Espaço para queda rápida",
            "Enter para pausar",
            "ESC para sair"
        ]
        prompt_text = "Pressione ENTER para começar"

    y_offset = SCREEN_HEIGHT // 2 - 100
    for line in instructions:
        render_text_with_border(screen, line, font, WHITE, BLACK, SCREEN_WIDTH // 2 - 150, y_offset)
        y_offset += 40
    render_text_with_border(screen, prompt_text, font, WHITE, BLACK, SCREEN_WIDTH // 2 - 150, y_offset + 20)
    pygame.display.flip()

# Function to display the language selection screen
def language_screen(screen):
    screen.blit(bg_language, (0, 0))
    font = pygame.font.SysFont('Comic Sans MS', 24)
    english_text = "Press 1 for English"
    portuguese_text = "Pressione 2 para Português"
    explanation_text = "Use the numbers to select the language"
    render_text_with_border(screen, explanation_text, font, WHITE, BLACK, SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 60)
    render_text_with_border(screen, english_text, font, WHITE, BLACK, SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 30)
    render_text_with_border(screen, portuguese_text, font, WHITE, BLACK, SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 10)
    pygame.display.flip()

# Main game loop
def main_game():
    clock = pygame.time.Clock()
    locked_positions = {}
    grid = create_grid(locked_positions)
    current_tetromino = Tetromino(random.choice(SHAPES), random.choice(SHAPE_COLORS))
    next_tetromino = Tetromino(random.choice(SHAPES), random.choice(SHAPE_COLORS))
    fall_time = 0
    fall_speed = 0.5
    score = 0
    font = pygame.font.SysFont('Comic Sans MS', 24)
    motivational_font = pygame.font.SysFont('Comic Sans MS', 20)
    motivational_message = ""
    motivational_message_time = 0
    running = True
    paused = False
    fast_drop = False
    game_start_time = pygame.time.get_ticks()

    left_pressed, right_pressed, down_pressed = False, False, False
    move_delay = 0

    while running:
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        clock.tick()

        if not paused:
            # Increase game speed over time (more gradual)
            fall_speed = max(0.3, 0.5 - (pygame.time.get_ticks() - game_start_time) / 180000)

            if fast_drop or fall_time / 1000 >= fall_speed:
                fall_time = 0
                current_tetromino.move(0, 1)
                if check_collision(current_tetromino, grid):
                    current_tetromino.move(0, -1)
                    for i, row in enumerate(current_tetromino.shape):
                        for j, cell in enumerate(row):
                            if cell:
                                locked_positions[(current_tetromino.x + j, current_tetromino.y + i)] = current_tetromino.color
                    current_tetromino = next_tetromino
                    next_tetromino = Tetromino(random.choice(SHAPES), random.choice(SHAPE_COLORS))
                    if check_collision(current_tetromino, grid):
                        game_over_screen(screen, score)
                        return
                    fast_drop = False

            # Handle continuous movement
            move_delay += clock.get_time()
            if move_delay > 150:  # Adjust the delay for left/right movement speed
                move_delay = 0
                if left_pressed:
                    current_tetromino.move(-1, 0)
                    if check_collision(current_tetromino, grid):
                        current_tetromino.move(1, 0)
                if right_pressed:
                    current_tetromino.move(1, 0)
                    if check_collision(current_tetromino, grid):
                        current_tetromino.move(-1, 0)
                if down_pressed:
                    current_tetromino.move(0, 1)
                    if check_collision(current_tetromino, grid):
                        current_tetromino.move(0, -1)

            rows_cleared = clear_rows(grid, locked_positions)
            if rows_cleared:
                score += rows_cleared * 100
                motivational_message = random.choice(PHRASES[language])
                motivational_message_time = pygame.time.get_ticks()

            if score >= 500:  # Game is considered beaten after clearing 5 rows (500 score)
                win_screen(screen)
                return

            screen.blit(bg_game, (0, 0))
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    pygame.draw.rect(screen, grid[y][x], pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            current_tetromino.draw(screen)

            # Draw score
            render_text_with_border(screen, f"Score: {score}", font, WHITE, BLACK, 10, 10)

            # Display motivational message
            if motivational_message and (pygame.time.get_ticks() - motivational_message_time < 3000):
                if (pygame.time.get_ticks() - motivational_message_time) // 500 % 2 == 0:
                    render_text_with_border(screen, motivational_message, motivational_font, WHITE, BLACK, SCREEN_WIDTH // 2 - motivational_font.size(motivational_message)[0] // 2, SCREEN_HEIGHT // 2)

            pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_LEFT:
                    left_pressed = True
                    if not paused:
                        current_tetromino.move(-1, 0)
                        if check_collision(current_tetromino, grid):
                            current_tetromino.move(1, 0)
                if event.key == pygame.K_RIGHT:
                    right_pressed = True
                    if not paused:
                        current_tetromino.move(1, 0)
                        if check_collision(current_tetromino, grid):
                            current_tetromino.move(-1, 0)
                if event.key == pygame.K_DOWN:
                    down_pressed = True
                if event.key == pygame.K_UP:
                    if not paused:
                        current_tetromino.rotate()
                        if check_collision(current_tetromino, grid):
                            current_tetromino.rotate()
                            current_tetromino.rotate()
                            current_tetromino.rotate()
                if event.key == pygame.K_SPACE:
                    fast_drop = True
                if event.key == pygame.K_RETURN:
                    paused = not paused
                    if paused:
                        pause_screen(screen)

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    left_pressed = False
                if event.key == pygame.K_RIGHT:
                    right_pressed = False
                if event.key == pygame.K_DOWN:
                    down_pressed = False

    pygame.quit()

# Main menu loop
def main_menu(screen):
    global language
    running = True
    show_instructions = False
    show_title = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if event.type == pygame.KEYDOWN:
                if not show_title and not show_instructions:
                    if event.key == pygame.K_1:
                        language = "en"
                        show_title = True
                    if event.key == pygame.K_2:
                        language = "pt"
                        show_title = True
                elif show_title and not show_instructions:
                    if event.key == pygame.K_RETURN:
                        show_instructions = True
                elif show_instructions:
                    if event.key == pygame.K_RETURN:
                        main_game()
                        return

        if not show_title and not show_instructions:
            language_screen(screen)
        elif show_title and not show_instructions:
            title_screen(screen)
        elif show_instructions:
            instructions_screen(screen)

    pygame.quit()

if __name__ == "__main__":
    main_menu(screen)
