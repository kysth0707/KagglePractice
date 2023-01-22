import dlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from PIL import Image

Detector = dlib.get_frontal_face_detector()
Predictor = dlib.shape_predictor(r'E:\GithubProjects\KagglePractice\HelmetDetection\models\shape_predictor_68_face_landmarks.dat')

def Rotate(ResultImage, angle):
	ResultImage = copy.deepcopy(ResultImage)
	image_center = tuple(np.array(ResultImage.shape[1::-1]) / 2)
	rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
	ResultImage = cv.warpAffine(ResultImage, rot_mat, ResultImage.shape[1::-1], flags=cv.INTER_LINEAR)
	return ResultImage

def ChangeToAvailable(value : int, maxValue : int):
	return 0 if value < 0 else maxValue if value > maxValue else value

def GetHelmetImage(TopPos, HelmetImagePos):
	ClearImage = None

	img = cv.imread(TopPos)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

	ALL = list(range(0, 68)) 
	RIGHT_EYEBROW = list(range(17, 22))  
	LEFT_EYEBROW = list(range(22, 27))  
	RIGHT_EYE = list(range(36, 42))  
	LEFT_EYE = list(range(42, 48))  
	NOSE = list(range(27, 36))  
	MOUTH_OUTLINE = list(range(48, 61))  
	MOUTH_INNER = list(range(61, 68)) 
	JAWLINE = list(range(0, 17)) 

	GrayImage = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

	dets = Detector(GrayImage)

	index = ALL

	ShowImage = copy.deepcopy(img)

	face = dets[0]


	shape = Predictor(ShowImage, face)

	list_points = []
	for p in shape.parts():
		list_points.append([p.x, p.y])
	list_points = np.array(list_points)

	for i,pt in enumerate(list_points[index]):

		pt_pos = (pt[0], pt[1])

	RightEyePosList = []
	for i, pt in enumerate(list_points[RIGHT_EYE]):
		RightEyePosList.append((pt[0], pt[1]))

	LeftEyePosList = []
	for i, pt in enumerate(list_points[LEFT_EYE]):
		LeftEyePosList.append((pt[0], pt[1]))

	LeftEyeCenter = (int(sum(map(lambda x : x[0], RightEyePosList)) / len(RightEyePosList)),
						int(sum(map(lambda x : x[1], RightEyePosList)) / len(RightEyePosList)))

	RightEyeCenter = (int(sum(map(lambda x : x[0], LeftEyePosList)) / len(LeftEyePosList)),
						int(sum(map(lambda x : x[1], LeftEyePosList)) / len(LeftEyePosList)))

	EyeWidth, EyeHeight = RightEyeCenter[0] - LeftEyeCenter[0], RightEyeCenter[1] - LeftEyeCenter[1]
	Radian = math.atan2(EyeHeight, EyeWidth)
	Angle = Radian * 180 / math.pi

	FaceWidth = int(face.width())
	WidthSlash2 = int(face.width() / 2)
	x0,y0 = face.left() - FaceWidth, face.top() - FaceWidth
	x1,y1 = face.right() + FaceWidth, face.bottom() + FaceWidth
	ImageHeight, ImageWidth = img.shape[:2]
	x0,y0,x1,y1 = ChangeToAvailable(x0,ImageWidth),ChangeToAvailable(y0,ImageHeight),ChangeToAvailable(x1,ImageWidth),ChangeToAvailable(y1,ImageHeight)
	RotatedImage = Rotate(img[y0:y1,x0:x1], Angle)

	ImageHeight, ImageWidth = RotatedImage.shape[:2]
	PaddingValueX = int(ImageWidth/8)
	PaddingValueY = int(ImageHeight/8)
	x0,y0,x1,y1=PaddingValueX,PaddingValueY,ImageWidth - PaddingValueX,ImageHeight - PaddingValueY
	RotatedImage = RotatedImage[y0:y1,x0:x1]

	HelmetImage = cv.imread(HelmetImagePos, -1)
	HelmetImage = cv.cvtColor(HelmetImage, cv.COLOR_BGR2RGB)

	Aindex = ALL

	ShowImage = copy.deepcopy(RotatedImage)
	ClearImage = copy.deepcopy(RotatedImage)

	GrayImage = cv.cvtColor(ShowImage, cv.COLOR_RGB2GRAY)
	dets = Detector(GrayImage)
	face = dets[0]

	shape = Predictor(ShowImage, face)

	list_points = []
	for p in shape.parts():
		list_points.append([p.x, p.y])
	list_points = np.array(list_points)

	for i,pt in enumerate(list_points[index]):

		pt_pos = (pt[0], pt[1])

	RightEyeBrowPosList = []
	for i, pt in enumerate(list_points[RIGHT_EYEBROW]):
		RightEyeBrowPosList.append((pt[0], pt[1]))

	LeftEyeBrowPosList = []
	for i, pt in enumerate(list_points[LEFT_EYEBROW]):
		LeftEyeBrowPosList.append((pt[0], pt[1]))

	LeftEyeBrowCenter = (int(sum(map(lambda x : x[0], RightEyeBrowPosList)) / len(RightEyeBrowPosList)),
						int(sum(map(lambda x : x[1], RightEyeBrowPosList)) / len(RightEyeBrowPosList)))

	RightEyeBrowCenter = (int(sum(map(lambda x : x[0], LeftEyeBrowPosList)) / len(LeftEyeBrowPosList)),
						int(sum(map(lambda x : x[1], LeftEyeBrowPosList)) / len(LeftEyeBrowPosList)))


	AddedImage = copy.deepcopy(RotatedImage)
	HelmetWidth = RightEyeBrowCenter[0] - LeftEyeBrowCenter[0]
	# Width : Height = HelmetWidth : ?
	# ? = HelmetWidth * HEight / Width
	# 100 x 57
	# ? = HelmetWidth * 57 / 100
	HelmetImageCopy = cv.resize(copy.deepcopy(HelmetImage), (int(HelmetWidth * 2.5), int(HelmetWidth * 0.57 * 2.5)))
	x_offset = int((RightEyeBrowCenter[0] + LeftEyeBrowCenter[0])/2 - HelmetWidth * 1.25)
	y_offset = int((RightEyeBrowCenter[1] + LeftEyeBrowCenter[1])/2 - HelmetWidth * 1.5)
	CheckArray = np.array([0, 0, 0])
	for y in range(HelmetImageCopy.shape[0]):
		for x in range(HelmetImageCopy.shape[1]):
			try:
				if y_offset+y < 0 or x_offset+x < 0:
					continue
				if HelmetImageCopy[y][x][0] == 255:
					AddedImage[y_offset+y,x_offset+x] = HelmetImageCopy[y][x]
				# else:
				# 	AddedImage[y_offset+y,x_offset+x] = np.array([255, 0, 0])
			except:
				pass

	return (AddedImage, ClearImage)