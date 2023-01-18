import pygame
from math import sqrt

MouseDrag = False
MousePos = []

def GetDistance(Pos1, Pos2):
	return sqrt((Pos1[0] - Pos2[0]) ** 2 + (Pos1[1] - Pos2[1]) ** 2)

pygame.init()

ScreenWidth = 900
ScreenHeight = 900
screen = pygame.display.set_mode((ScreenWidth, ScreenHeight))
pygame.display.set_caption('숫자를 그려주세요 0 ~ 9')

clock = pygame.time.Clock()
Run = True
while Run:
	clock.tick(60)

	screen.fill((255, 255, 255))

	if MouseDrag:
		if len(MousePos) <= 2:
			MousePos.append(pygame.mouse.get_pos())
		elif GetDistance(pygame.mouse.get_pos(), MousePos[len(MousePos)-1]) > 20:
			MousePos.append(pygame.mouse.get_pos())
	for i in range(len(MousePos) - 1):
		pygame.draw.line(screen, (0, 0, 0), MousePos[i], MousePos[i + 1], width = 5)

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			Run = False
			pygame.quit()
			break

		elif event.type == pygame.MOUSEBUTTONDOWN:
			if event.button == 1:
				MouseDrag = True
		
		elif event.type == pygame.MOUSEBUTTONUP:
			if event.button == 1:
				MouseDrag = False

	pygame.display.update()