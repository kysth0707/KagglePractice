import numpy as np
import tensorflow as tf
from tkinter.filedialog import *


def ReturnPos(loc):
	return "E:\\GithubProjects\\KagglePractice\\GamePlay" + str(loc)

ModelDir = ReturnPos("\\Model")
ImageSize = (640, 360)

model = tf.keras.models.load_model(ModelDir)

# ImageLoc = ReturnPos("\\Dataset\\Test\\Among Us\\image_14.png")
# ImageLoc = ReturnPos("\\Dataset\\Test\\Minecraft\\image_18.png")


while True:
	ImageLoc = askopenfilename(initialdir=ReturnPos("\\Dataset\\Test"), title="이미지 선택", filetypes=(("png 파일", "*.png"), ("모든 파일", "*.*")))
	Image = tf.keras.utils.load_img(ImageLoc, target_size = ImageSize)

	x = tf.keras.utils.img_to_array(Image)
	x = np.expand_dims(x, axis=0)
	predict = model.predict([x])

	if predict[0] > 0.5:
		print("Minecraft")
	else:
		print("Among Us")