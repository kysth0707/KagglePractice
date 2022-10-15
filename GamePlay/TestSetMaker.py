import os
import shutil
import random

def ReturnPos(loc):
	return "E:\\GithubProjects\\KagglePractice\\GamePlay" + str(loc)

ImageDirs = {
			"Roblox" : ReturnPos("\\Dataset\\Roblox"),
			"Terraria" : ReturnPos("\\Dataset\\Terraria"),
}

Status = 0
TestPercent = 0.2
for key in ImageDirs:
	Datas = os.listdir(ImageDirs[key])

	random.shuffle(Datas)
	for i in range(int(TestPercent * len(Datas))):
		Image = Datas.pop()
		try:
			shutil.move(ImageDirs[key] + "\\" + Image, ImageDirs[key] + "\\Test\\" + Image)
		except:
			pass
	for Image in Datas:
		try:
			shutil.move(ImageDirs[key] + "\\" + Image, ImageDirs[key] + "\\Learn\\" + Image)
		except:
			pass

	Status += 1
	if Status % 100 == 0:
		print(Status)