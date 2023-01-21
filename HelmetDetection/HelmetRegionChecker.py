import torch
import torchvision
from torchvision import models
import torchvision.transforms as T

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy

from matplotlib.path import Path
import matplotlib.patches as patches

print('pytorch', torch.__version__)
print('torchvision', torchvision.__version__)

# GetHelmetImages(r'E:\GithubProjects\KagglePractice\HelmetDetection\TestImage\test.jpg')

ColorArray = np.array([
	(0, 0, 0),       # 0=background
	(128, 0, 0),     # 1=aeroplane
	(0, 128, 0),     # 2=bicycle
	(128, 128, 0),   # 3=bird
	(0, 0, 128),     # 4=boat
	(128, 0, 128),   # 5=bottle
	(0, 128, 128),   # 6=bus
	(128, 128, 128), # 7=car
	(255, 255, 127), # 8=cat
	(192, 0, 0),     # 9=chair
	(64, 128, 0),    # 10=cow
	(192, 128, 0),   # 11=dining table
	(64, 0, 128),    # 12=dog
	(192, 0, 128),   # 13=horse
	(64, 128, 128),  # 14=motorbike
	(255, 255, 255), # 15=person
	(0, 64, 0),      # 16=potted plant
	(128, 64, 0),    # 17=sheep
	(0, 192, 0),     # 18=sofa
	(128, 192, 0),   # 19=train
	(0, 64, 128)     # 20=tv/monitor
])

def seg_map(img, n_classes=21):
	global ColorArray

	rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

	for c in range(n_classes):
		idx = img == c

		rgb[idx] = ColorArray[c]

	return rgb

def GetHelmetImages(ImagePos):
	"""
	이미지 주소를 입력해주세요
	"""
	# Section1
	IMG_SIZE = 480

	deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
	img = Image.open(ImagePos)

	trf = T.Compose([
		T.Resize(IMG_SIZE),
		T.ToTensor(),
		T.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
	])

	input_img = trf(img).unsqueeze(0)

	out = deeplab(input_img)['out']

	out = torch.argmax(out.squeeze(), dim=0)
	out = out.detach().cpu().numpy()

	out_seg = seg_map(out)
	ClearImage = img.resize((640, 480)) & out_seg

	# Section 2

	THRESHOLD = 0.95
	
	model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

	ClearImage = Image.fromarray(ClearImage)
	img = ClearImage.resize((IMG_SIZE, int(ClearImage.height * IMG_SIZE / ClearImage.width)))

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

	# fig, ax = plt.subplots(1, figsize=(16, 16))
	# ax.imshow(img)

	HeadList = []

	for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
		score = score.detach().numpy()

		if score < THRESHOLD:
			continue

		box = box.detach().numpy()
		keypoints = keypoints.detach().numpy()[:, :2]

		# rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='b', facecolor='none')
		# ax.add_patch(rect)
		x0, y0, x1, y1 = box
		# ax.add_patch(patches.Circle((x0, y0), radius=2, facecolor='c'))
		# ax.add_patch(patches.Circle((x1, y0), radius=2, facecolor='c'))

		HeadPointLocs = []
		for k in keypoints[:5]:
			# circle = patches.Circle((k[0], k[1]), radius=2, facecolor='r')
			HeadPointLocs.append((k[0], k[1]))
			# ax.add_patch(circle)
		
		HeadMiddlePoint = (sum(map(lambda x : x[0], HeadPointLocs)) / len(HeadPointLocs),
						sum(map(lambda x : x[1], HeadPointLocs)) / len(HeadPointLocs))
		# ax.add_patch(patches.Circle(HeadMiddlePoint, radius=2, facecolor='g'))

		# rect = patches.Rectangle((x0, y0), x1 - x0, (HeadMiddlePoint[1] - y0) * 2, linewidth=2, edgecolor='y', facecolor='none')
		# ax.add_patch(rect)
		HeadList.append((
						np.asarray(img)[int(y0):int(y0 + (HeadMiddlePoint[1] - y0) * 2),int(x0) : int(x1)],
						(int(x0), int(y0)),
						(int(x1), int(y0 + (HeadMiddlePoint[1] - y0) * 2))
						))
	
	return HeadList

def ShowHelmetImages(HeadList):
	for i, img in enumerate(HeadList):
		plt.subplot(1, 5, i + 1)
		plt.imshow(img[0])
	plt.show()