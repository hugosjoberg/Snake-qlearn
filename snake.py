import sys, pygame, random, itertools, time

class Game(object):

    def __init__(self):
        self.screen = pygame.display.set_mode((250, 250))

        self.scr_width = self.screen.get_rect().width
        self.scr_height = self.screen.get_rect().height
        self.size = self.screen.get_size()
        self.grid_square = 10
        self.keys_pressed = []
        self.set_start_state()
        self.score = 0

    def set_start_state(self):
        width, height = self.size
        self.screen_center = [width / 2, height / 2]
        self.snake = Snake(self.screen_center)
        self.food = Food(self.new_food_position())

    def run(self,actions):

        pygame.event.pump()
        #self.keys_pressed = pygame.key.get_pressed()
        self.update(actions)
        if self.snake_out_of_bounds() or self.snake.self_collision():
            print('The game ended')
            print('The score was: ',self.score)
            pygame.QUIT
            sys.exit(1)

        self.draw()
        time.sleep(80.0/1000.0)
        image = pygame.surfarray.array3d(pygame.display.get_surface())

        return image
    def update(self,actions):
        food_collision = self.food.position == self.snake.positions[0]
        if food_collision:
            self.score+=1
            self.food.position = self.new_food_position()
            print(self.score)
        self.snake.update(self.get_direction(actions), food_collision)

    def get_direction(self,action):

        if action==0:
            return [-self.grid_square, 0]
        elif action==1:
            return [self.grid_square, 0]
        elif action==2:
            return [0, -self.grid_square]
        elif action==3:
            return [0, self.grid_square]
        else:
            return [0, 0]

    def new_food_position(self):
        x = self.random_valid_coordinate(0)
        y = self.random_valid_coordinate(1)
        for position in self.snake.positions:
            if [x, y] == position:
                return self.new_food_position()
        return [x, y]

    def random_valid_coordinate(self, axis):
        num_squares = self.size[axis] / self.grid_square - 1
        return random.randint(0, num_squares) * self.grid_square

    def snake_out_of_bounds(self):
        head_posx, head_posy = self.snake.positions[0]
        return head_posx < 0 or head_posx > self.size[0] - self.grid_square or\
               head_posy < 0 or head_posy > self.size[1] - self.grid_square

    def draw(self):
        self.screen.fill((0,0,0))
        self.food.draw(self.screen)
        self.snake.draw(self.screen)
        pygame.display.flip()

class Snake(object):
    def __init__(self, position, speed=[0,0]):
        self.positions = [position]
        self.speed = speed

    def update(self, speed_delta, food_collision=False):
        self.set_speed(speed_delta)
        head_pos = self.new_head_position()

        self.positions.insert(0, head_pos)
        if not food_collision:
            self.positions.pop()

    def new_head_position(self):
        current_head = self.positions[0]
        return [current_head[0] + self.speed[0],
                current_head[1] + self.speed[1]]

    def set_speed(self, speed_delta):
        if speed_delta == [0, 0]:
            pass
        elif self.speed == [0, 0]:
            self.speed = speed_delta
        # change speed only if speed_delta is orthogonal to speed
        elif abs(self.speed[0]) != abs(speed_delta[0]):
            self.speed = speed_delta

    def self_collision(self):
        head_pos = self.positions[0]
        for position in self.positions[1:]:
            if position == head_pos:
                return True
        return False

    def draw(self, screen):
        square = pygame.Surface((10, 10))
        square.fill((0, 255, 0))
        for position in self.positions:
            screen.blit(square, position)

class Food(object):
    def __init__(self, position):
        self.square = pygame.Surface((10, 10))
        self.position = position

    def draw(self,screen):
        self.square.fill((0, 0, 255))
        screen.blit(self.square,self.position)
'''
def runner():
    screen = pygame.display.set_mode((210, 160))

    game = Game(screen)

    while True:

        game.run()
        #game.set_start_state()



if __name__ == "__main__":
    pygame.init()
    runner()
'''
