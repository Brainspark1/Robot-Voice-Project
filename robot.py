import pygame
import sys
from inference import predict_command

pygame.init()

WIDTH, HEIGHT = 600, 600
GRID_SIZE = 20

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot")

clock = pygame.time.Clock()

x = WIDTH // 2
y = HEIGHT // 2
step = 20

# note that 0=up, 1=right, 2=down, 3=left
direction = 0

def move_robot(command):
    global x, y, direction

    if command == "go":
        if direction == 0:
            y -= step
        elif direction == 1:
            x += step
        elif direction == 2:
            y += step
        elif direction == 3:
            x -= step

    elif command == "left":
        direction = (direction - 1) % 4

    elif command == "right":
        direction = (direction + 1) % 4

    elif command == "stop":
        pass


def draw():
    screen.fill((30, 30, 30))

    # grid
    for i in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, (50, 50, 50), (i, 0), (i, HEIGHT))
    for j in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, (50, 50, 50), (0, j), (WIDTH, j))

    # robot (dot)
    pygame.draw.circle(screen, (0, 255, 0), (x, y), 10)

    pygame.display.update()


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    label, conf = predict_command()

    if conf > 0.75:
        move_robot(label)

    draw()
    clock.tick(10)