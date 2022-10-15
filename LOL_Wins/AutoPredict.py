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



DataY = dataframe['blueWins'].values 
DataX = []
for i, rows in dataframe.iterrows():
	Temp = []
	for i in range(DataHeaderLen):
		Temp.append(rows[DataHeaders[i]])
	DataX.append(Temp)


# 테스트 셋, 트레인 셋 설정
from sklearn.model_selection import train_test_split
TrainX, TestX, TrainY, TestY = train_test_split(DataX, DataY, test_size=0.2, random_state=42)

# 정규화
TestX = np.array(TestX)
TestX = TestX / TestX.max()

model = tf.keras.models.load_model(ModelLoc)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])


predict = model.predict(TrainX)
PercentA = sum(predict) / len(predict) * 100

predict = model.predict(TestX)
PercentB = sum(predict) / len(predict) * 100

PercentC = dataframe['blueWins'].sum() / dataframe['blueWins'].count() * 100

print(f"트레인 셋 예측 확률 : {PercentA} %")
print(f"테스트 셋 예측 확률 : {PercentB} %")
print(f"실제 확률 : {PercentC}")

print(f"\n\n데이터 기준 : 약 {PercentA / PercentB * 100} %")