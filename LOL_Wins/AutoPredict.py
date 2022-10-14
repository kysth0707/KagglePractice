import tensorflow as tf
import pandas as pd
import numpy as np

dataframe = pd.read_csv("E:\\GithubProjects\\KagglePractice\\LOL_Wins\\high_diamond_ranked_10min.csv")
ModelLoc = "E:\\GithubProjects\\KagglePractice\\LOL_Wins\\Model"



# print(list(dataframe.columns.values))
DataHeaderDict = {'blueWardsPlaced' : False,
				'blueWardsDestroyed' : False, 
				'blueFirstBlood' : True, 
				'blueKills' : True, 
  				'blueDeaths' : True, 
  				'blueAssists' : True, 
  				'blueEliteMonsters' : False, 
  				'blueDragons' : True, 
  				'blueHeralds' : True, 
  				'blueTowersDestroyed' : False, 
  				'blueTotalGold' : True, 
  				'blueAvgLevel' : True, 
  				'blueTotalExperience' : False, 
  				'blueTotalMinionsKilled' : True, 
  				'blueTotalJungleMinionsKilled' : False, 
  				'blueGoldDiff' : True, 
  				'blueExperienceDiff' : True, 
  				'blueCSPerMin' : True, 
  				'blueGoldPerMin' : True}


DataHeaders = []
for key in DataHeaderDict:
	if DataHeaderDict[key] == True:
		DataHeaders.append(key)
print(DataHeaders)

DataHeaderLen = len(DataHeaders)



TestY = dataframe['blueWins'].values 
TestX = []
for i, rows in dataframe.iterrows():
	Temp = []
	for i in range(DataHeaderLen):
		Temp.append(rows[DataHeaders[i]])
	TestX.append(Temp)


# 정규화
TestX = np.array(TestX)
TestX = TestX / TestX.max()


model = tf.keras.models.load_model(ModelLoc)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])


predict = model.predict(TestX)
FullPercent = 0
for Data in predict:
	FullPercent += Data[0]
	
PercentA = FullPercent / len(predict) * 100
print(f"예측 확률 : {PercentA} %")

PercentB = dataframe['blueWins'].sum() / dataframe['blueWins'].count() * 100
print(f"실제 확률 : {PercentB} %")

print(f"\n\n전체 데이터 기준 : 약 {PercentA / PercentB * 100} %")