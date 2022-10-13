import tensorflow as tf
import pandas as pd
import numpy as np
# from keras.optimizers import SGD
import time

dataframe = pd.read_csv("E:\\GithubProjects\\KagglePractice\\University\\gpascore.csv")
ModelLoc = "E:\\GithubProjects\\KagglePractice\\University\\Model"

dataframe = dataframe.dropna()

ModelAlreayHave = True


# print(list(dataframe.columns.values))

print("Loading datas by csv..")
lasttime = time.time()

DataY = dataframe['admit'].values
DataX = []
for i, rows in dataframe.iterrows():
	DataX.append([rows['gre'], rows['gpa'], rows['rank']])

print(f"Loaded well - {time.time() - lasttime} sec")





if ModelAlreayHave:
	model = tf.keras.models.load_model(ModelLoc)
else:
	# https://stackoverflow.com/questions/37213388/keras-accuracy-does-not-change
	# sigmoid 가 정확도 잘 올라감
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(64, activation='sigmoid'),
		tf.keras.layers.Dense(128, activation='sigmoid'),
		tf.keras.layers.Dense(1, activation='sigmoid'), #sigmoid : 0~1 사이 결과 값으로 바꿔줌
		# 내가 원하는 값이 0~1 이면 마지막 레이어 1
	])



# opt = SGD(lr=0.01)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# model.fit(학습, 결과, epochs = 학습횟수)
model.fit(np.array(DataX), np.array(DataY), epochs = 1000)


model.save(ModelLoc)