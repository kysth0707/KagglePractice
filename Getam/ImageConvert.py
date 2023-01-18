# 
# cv.imread 를 하기 위해서는 경로가 '영어, 숫자' 여야 합니다. 한국어 안돼요!
# 

import cv2 as cv
import numpy as np
import os

def ImageShow(Image):
	cv.imshow('Image', Image)
	cv.waitKey()
	cv.destroyAllWindows()

def ImageDown(Image):
	RemoveStruct = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
	return cv.erode(Image, RemoveStruct)

def ImageUp(Image):
	RemoveStruct = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
	return cv.dilate(Image, RemoveStruct)

def GetItems(TopPos):
	global Index
	for file in os.listdir(TopPos):
		TestImage = cv.imread(fr'{TopPos}\{file}')
		Height, Width = TestImage.shape[0], TestImage.shape[1]
		# print(Width, Height)
		TestImage = cv.cvtColor(TestImage, cv.COLOR_BGR2GRAY)
		# TestImage = cv.GaussianBlur(TestImage, (5, 5), 0)
		_, TestImage = cv.threshold(TestImage, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

		TestImage = ImageDown(TestImage)
		# for _ in range(2):
		# TestImage = ImageUp(TestImage)
		# TestImage = ImageDown(TestImage)

		FileText = file.split('.')[0]
		for i in range(5):
			x1 = int(int((Width-14) / 5) * i) + 7
			x2 = int(int((Width-14) / 5) * (i + 1)) + 7
			cv.imwrite(rf'E:\GithubProjects\KagglePractice\Getam\Dataset\{FileText[i]}-{Index}.bmp', TestImage[:Height,x1:x2])
			# print(FileText[i])

			Index += 1

Index = 0
GetItems(r'E:\GithubProjects\KagglePractice\Getam\RawDataset\new-1')
GetItems(r'E:\GithubProjects\KagglePractice\Getam\RawDataset\new-2')
GetItems(r'E:\GithubProjects\KagglePractice\Getam\RawDataset\new-3')