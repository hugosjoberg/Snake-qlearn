import pygame
import random
import time
import math
import os


class Game(object):
    def __init__(self):
        pygame.init()
        self.height = 400
        self.width = 400
        self.border_width = 10
        self.s = pygame.display.set_mode((self.height, self.width))
        self.speed = 20
        self.set_start_state()

    def set_start_state(self):
        self.snake_size = 20
        self.food_size = 20
        self.xs = [
            self.width / 2 - self.snake_size,
            self.width / 2 - self.snake_size,
            self.width / 2 - self.snake_size,
            self.width / 2 - self.snake_size,
        ]
        self.ys = [
            self.height / 2 - self.snake_size,
            self.height / 2 - 2 * self.snake_size,
            self.height / 2 - 3 * self.snake_size,
            self.height / 2 - 4 * self.snake_size,
        ]
        self.dirs = random.choice([0, 1, 2, 3])
        self.score = 0
        self.applepos = (
            random.randint(
                self.border_width + self.food_size,
                self.height - self.food_size - self.border_width,
            ),
            random.randint(
                self.border_width + self.food_size,
                self.width - self.food_size - self.border_width,
            ),
        )
        self.appleimage = pygame.Surface((self.food_size, self.food_size))
        self.appleimage.fill((0, 255, 0))
        self.img = pygame.Surface((self.food_size, self.food_size))
        self.img.fill((0, 0, 0))
        self.game_over = False
        self.new_distance = 0
        self.old_distance = 0

    def distance(self, x1, x2, y1, y2):
        x = math.pow((x1 - x2), 2)
        y = math.pow((y1 - y2), 2)
        distance = math.sqrt(x + y)
        return distance

    def collide(self, x1, x2, y1, y2, w1, w2, h1, h2):
        if x1 + w1 > x2 and x1 < x2 + w2 and y1 + h1 > y2 and y1 < y2 + h2:
            return True
        else:
            return False

    def reward(self, apple_eaten):
        if self.new_distance < self.old_distance:
            reward = 0.4
        else:
            reward = -0.4
        if apple_eaten:
            reward = 1.0
        return reward

    def die(self):
        reward = -1
        self.draw_board()
        image = pygame.surfarray.array3d(pygame.display.get_surface())
        time.sleep(1 / 10)
        return image, reward, True

    def direction_snake(self, actions):
        action = actions
        if action == 2 and self.dirs != 0:
            self.dirs = 2
        elif action == 0 and self.dirs != 2:
            self.dirs = 0
        elif action == 3 and self.dirs != 1:
            self.dirs = 3
        elif action == 1 and self.dirs != 3:
            self.dirs = 1
        dirs = self.dirs
        self.move_snake(dirs)

    def move_snake(self, dirs):
        if dirs == 0:
            self.ys[0] += self.speed
        elif dirs == 1:
            self.xs[0] += self.speed
        elif dirs == 2:
            self.ys[0] -= self.speed
        elif dirs == 3:
            self.xs[0] -= self.speed

    def draw_board(self):
        self.s.fill((255, 255, 255))
        for i in range(0, len(self.xs)):
            self.s.blit(self.img, (self.xs[i], self.ys[i]))
        self.s.blit(self.appleimage, self.applepos)
        pygame.draw.rect(self.s, (0, 0, 0), [0, 0, self.width, self.border_width])
        pygame.draw.rect(
            self.s,
            (0, 0, 0),
            [0, self.height - self.border_width, self.width, self.border_width],
        )
        pygame.draw.rect(self.s, (0, 0, 0), [0, 0, self.border_width, self.height])
        pygame.draw.rect(
            self.s,
            (0, 0, 0),
            [self.width - self.border_width, 0, self.border_width, self.height],
        )
        pygame.display.update()

    def run(self, actions):
        actions = actions
        i = len(self.xs) - 1
        pygame.event.pump()
        snake_eat_apple = False
        self.direction_snake(actions)
        while i >= 1:
            self.xs[i] = self.xs[i - 1]
            self.ys[i] = self.ys[i - 1]
            i -= 1
        i = len(self.xs) - 1
        # Check if snake collide with self
        while i >= 2:
            if self.collide(
                self.xs[0],
                self.xs[i],
                self.ys[0],
                self.ys[i],
                self.snake_size,
                self.snake_size,
                self.snake_size,
                self.snake_size,
            ):
                return self.die()
            i -= 1

        # Check if snake collide with apple
        if self.collide(
            self.xs[0],
            self.applepos[0],
            self.ys[0],
            self.applepos[1],
            self.snake_size,
            self.food_size,
            self.snake_size,
            self.food_size,
        ):
            self.score += 1
            self.xs.append(700)
            self.ys.append(700)
            self.applepos = (
                random.randint(
                    self.border_width + self.food_size,
                    self.height - self.food_size - self.border_width,
                ),
                random.randint(
                    self.border_width + self.food_size,
                    self.width - self.food_size - self.border_width,
                ),
            )
            snake_eat_apple = True

        # Check if snake collide with wall
        if (
            self.xs[0] < self.border_width / 2
            or self.xs[0] > self.width - self.snake_size - self.border_width / 2
            or self.ys[0] < self.border_width / 2
            or self.ys[0] > self.height - self.snake_size - self.border_width / 2
        ):
            return self.die()

        # Calculate new distance between snake head and food
        self.old_distance = self.new_distance
        self.new_distance = self.distance(
            self.xs[0], self.applepos[0], self.ys[0], self.applepos[1]
        )
        self.draw_board()
        time.sleep(1 / 20)
        image = pygame.surfarray.array3d(pygame.display.get_surface())
        return image, self.reward(snake_eat_apple), self.game_over
