import pygame, random, sys, time
from pygame.locals import *

class Game(object):

	def __init__(self):
		pygame.init()
		self.s = pygame.display.set_mode((600, 600))
		pygame.display.set_caption('Snake')
		self.f = pygame.font.SysFont('Arial', 20)

		self.set_start_state()

	def set_start_state(self):
		self.xs = [290, 290]
		self.ys = [290, 270]
		self.dirs = 4
		self.score = 0
		self.applepos = (random.randint(0, 590), random.randint(0, 590))
		self.appleimage = pygame.Surface((10, 10))
		self.appleimage.fill((0, 255, 0))
		self.img = pygame.Surface((20, 20))
		self.img.fill((255, 0, 0))
		self.game_over = False

	def collide(self,x1, x2, y1, y2, w1, w2, h1, h2):
		if x1+w1>x2 and x1<x2+w2 and y1+h1>y2 and y1<y2+h2:
			return True
		else:
			return False

	def die(self):

		self.f=pygame.font.SysFont('Arial', 30)
		self.t=self.f.render('Your score was: '+str(self.score), True, (0, 0, 0))
		self.s.blit(self.t, (10, 270))
		pygame.display.update()
		self.game_over = True
		self.score = -1
		image = pygame.surfarray.array3d(pygame.display.get_surface())
		final_score = self.score
		game_over_flag = self.game_over
		time.sleep(0.5)
		self.__init__()
		return image, final_score, game_over_flag

	def direction_snake(self,actions):
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

	def move_snake(self,dirs):
		if dirs==0:
			self.ys[0] += 20
		elif dirs==1:
			self.xs[0] += 20
		elif dirs==2:
			self.ys[0] -= 20
		elif dirs==3:
			self.xs[0] -= 20


	def run(self,actions):
		actions = actions
		pygame.event.pump()
		self.direction_snake(actions)

		i = len(self.xs)-1
		while i >= 2:
			if self.collide(self.xs[0], self.xs[i], self.ys[0], self.ys[i], 20, 20, 20, 20):
				self.die()
			i-= 1

		if self.collide(self.xs[0], self.applepos[0], self.ys[0], self.applepos[1], 20, 10, 20, 10):
			self.score += 1
			self.xs.append(700)
			self.ys.append(700)
			self.applepos = (random.randint(0,590),random.randint(0,590))

		if self.xs[0] < 0 or self.xs[0] > 580 or self.ys[0] < 0 or self.ys[0] > 580:
			self.die()

		i = len(self.xs)-1
		while i >= 1:
			self.xs[i] = self.xs[i-1]
			self.ys[i] = self.ys[i-1]
			i -= 1

		self.s.fill((255, 255, 255))

		for i in range(0, len(self.xs)):
			self.s.blit(self.img, (self.xs[i], self.ys[i]))

		self.s.blit(self.appleimage, self.applepos)
		self.t=self.f.render(str(self.score), True, (0, 0, 0))
		self.s.blit(self.t, (10, 10))
		#pygame.display.update()
		pygame.display.flip()
		time.sleep(1/7)
		image = pygame.surfarray.array3d(pygame.display.get_surface())
		return image, self.score, self.game_over
