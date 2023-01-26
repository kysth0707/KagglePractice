from Module.ImagePrediction import GetPrediction
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Module.HeadRegion import GetHeadDict
from keras.models import load_model
import numpy as np


TestVideoPos = r'E:\GithubProjects\KagglePractice\HelmetDetection2\TestVideo.mp4'
VideoSavePos = r'E:\GithubProjects\KagglePractice\HelmetDetection2\output.mp4'
model = load_model(r'E:\GithubProjects\KagglePractice\HelmetDetection2\HelmetModel.h5')

cap = cv.VideoCapture(TestVideoPos)
ret, img = cap.read()
img = cv.resize(img, (480, int(480 * img.shape[0] / img.shape[1])))

fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv.VideoWriter(VideoSavePos, fourcc, cap.get(cv.CAP_PROP_FPS), (img.shape[1], img.shape[0]))

while cap.isOpened():
	ret, img = cap.read()
	if not ret:
		break
	h, w = img.shape[:2]

	result_img = img.copy()

	HeadDict = GetHeadDict(result_img, (100, 100), 10)

	Height, Width = result_img.shape[:2]
	result_img = cv.resize(result_img, (480, int(480 * Height / Width)))
	# ax.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))

	for head in HeadDict['headlist']:
		Prediction = model.predict(np.array(
			[head['head']]
		))

		Color = (0, 0, 255) if float(Prediction[0][0]) < 0.5 else (0, 255, 0)
		x0, y0, x1, y1 = head['bodypos']
		# rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor=Color, facecolor='none')
		# ax.add_patch(rect)
		cv.rectangle(result_img, pt1=(x0, y0), pt2=(x1, y1), thickness=2, color=Color, lineType=cv.LINE_AA)

		# ax.text(x=x0,
		# 		y=y0-5,
		# 		s=f"{int(Prediction * 100)} %",
		# 		color="black")
		cv.putText(result_img, text=f"{int(Prediction * 100)} %", org=(x0, y0-5),
				   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 0, 0), thickness=1, lineType=cv.LINE_AA)

	out.write(result_img)
	cv.imshow('result', result_img)
	if cv.waitKey(1) == ord('q'):
		break

out.release()
cap.release()

