import torch
import torchvision
from torchvision import models
import torchvision.transforms as T

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import cv2 as cv
import copy

print('pytorch', torch.__version__)
print('torchvision', torchvision.__version__)

model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

def GetHeadDict(img, NORMALIZE_SIZE : tuple = (-1, -1), TargetSize : int = 4000):
	# img = cv.imread(r'E:\GithubProjects\KagglePractice\HelmetDetection2\RawDatasets\pexels-aleksey-3680959.jpg')
	Height, Width = img.shape[:2]
	# Width : Height = 480 : ?
	# ? = 480 * Height / Width
	img = cv.resize(img, (480, int(480 * Height / Width)))
	originalimg = copy.deepcopy(img)

	THRESHOLD = 0.95
	OFFSET = 5
	# NORMALIZE_SIZE = (100, 100)
	# NORMALIZE_SIZE = (-1, -1)

	trf = T.Compose([
		T.ToTensor()
	])

	input_img = trf(img)

	out = model([input_img])[0]

	codes = [
		Path.MOVETO,
		Path.LINETO,
		Path.LINETO
	]

	HeadDict = {}
	HeadDict['originalimg'] = img.copy()
	HeadList = []

	for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
		score = score.detach().numpy()

		if score < THRESHOLD: # 정확도가 낮으면 사람 아님
			continue
		BoxSize = abs((box[2] - box[0]) * (box[3] - box[1]))
		if BoxSize < TargetSize: # 상자 크기가 4000 보다 작으면 사람 아닌 것으로 치부함
			continue

		box = box.detach().numpy()
		keypoints = keypoints.detach().numpy()[:, :2]
		x0, y0, x1, y1 = box

		# 머리 쪽만 확인
		HeadPointLocs = []
		for k in keypoints[:5]:
			HeadPointLocs.append((k[0], k[1]))
		LeftEarX = int(keypoints[4][0])
		RightEarX = int(keypoints[3][0])
		
		# 머리 키 포인트들의 중심점을 구합니다
		HeadMiddlePoint = (sum(map(lambda x : x[0], HeadPointLocs)) / len(HeadPointLocs),
						sum(map(lambda x : x[1], HeadPointLocs)) / len(HeadPointLocs))

		MiddleX,MiddleY = int((LeftEarX + RightEarX)/2), int(HeadMiddlePoint[1])
		RadiusValue = int(OFFSET + (HeadMiddlePoint[1] - y0))
		x0,x1 = MiddleX - RadiusValue, MiddleX + RadiusValue
		y0,y1 = MiddleY - RadiusValue, MiddleY + RadiusValue
		# 머리 부분을 구합니다

		rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='y', facecolor='none')

		HeadImage = originalimg[y0:y1,x0:x1]
		if NORMALIZE_SIZE != (-1, -1):
			HeadImage = cv.resize(HeadImage, NORMALIZE_SIZE)
		else:
			pass
		HeadList.append({
			"head" : HeadImage,
			"headpos" : (x0, y0, x1, y1),
			"bodypos" : (int(box[0]),int(box[1]),int(box[2]),int(box[3]))
		})
	HeadDict['headlist'] = HeadList
	return HeadDict