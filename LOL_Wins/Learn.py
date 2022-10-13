import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os

dataframe = pd.read_csv("E:\\GithubProjects\\KagglePractice\\LOL_Wins\\high_diamond_ranked_10min.csv")
ModelLoc = "E:\\GithubProjects\\KagglePractice\\LOL_Wins\\Model"
InFolder = os.listdir(ModelLoc)


ModelAlreayHave = True


# print(list(dataframe.columns.values))
DataHeaderDict = {'blueWardsPlaced' : False,
				'blueWardsDestroyed' : False, 
				'blueFirstBlood' : True, 
				'blueKills' : True, 
  				'blueDeaths' : True, 
  				'blueAssists' : False, 
  				'blueEliteMonsters' : False, 
  				'blueDragons' : True, 
  				'blueHeralds' : False, 
  				'blueTowersDestroyed' : False, 
  				'blueTotalGold' : True, 
  				'blueAvgLevel' : False, 
  				'blueTotalExperience' : False, 
  				'blueTotalMinionsKilled' : True, 
  				'blueTotalJungleMinionsKilled' : False, 
  				'blueGoldDiff' : False, 
  				'blueExperienceDiff' : False, 
  				'blueCSPerMin' : False, 
  				'blueGoldPerMin' : False}
DataHeaders = []
for key in DataHeaderDict:
	if DataHeaderDict[key] == True:
		DataHeaders.append(key)
print(DataHeaders)

DataHeaderLen = len(DataHeaders)

print("Loading datas by csv..")
lasttime = time.time()

DataY = dataframe['blueWins'].values 
DataX = []
for i, rows in dataframe.iterrows():
	Temp = []
	for i in range(DataHeaderLen):
		Temp.append(rows[DataHeaders[i]])
	DataX.append(Temp)
print(f"Loaded well - {time.time() - lasttime} sec")



if ModelAlreayHave:
	model = tf.keras.models.load_model(ModelLoc)
else:
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(64, activation='tanh'),
		tf.keras.layers.Dense(128, activation='tanh'),
		tf.keras.layers.Dense(1, activation='sigmoid'), #sigmoid : 0~1 사이 결과 값으로 바꿔줌
		# 내가 원하는 값이 0~1 이면 마지막 레이어 1
	])



model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# model.fit(학습, 결과, epochs = 학습횟수)
model.fit(np.array(DataX), np.array(DataY), epochs = 1000)


model.save(ModelLoc)