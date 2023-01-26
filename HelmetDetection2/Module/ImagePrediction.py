import cv2 as cv
import matplotlib.pyplot as plt
from Module.HeadRegion import GetHeadDict
from keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.patches as patches

model = load_model(r'E:\GithubProjects\KagglePractice\HelmetDetection2\HelmetModel.h5')

def GetPrediction(img):
	HeadDict = GetHeadDict(img, (100, 100), 10)

	fig, ax = plt.subplots(1, figsize=(16, 16))
	Height, Width = img.shape[:2]
	img = cv.resize(img, (480, int(480 * Height / Width)))
	im = ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

	for head in HeadDict['headlist']:
		Prediction = model.predict(np.array(
			[head['head']]
		))

		Color = "red" if float(Prediction[0][0]) < 0.5 else "green"
		x0, y0, x1, y1 = head['bodypos']
		rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor=Color, facecolor='none')
		ax.add_patch(rect)

		ax.text(x=x0,
				y=y0-5,
				s=f"{int(Prediction * 100)} %",
				color="black")
	return im