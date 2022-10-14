import tensorflow as tf
import pandas as pd

FalseValue = -12345

dataframe = pd.read_csv("E:\\GithubProjects\\KagglePractice\\LOL_Wins\\high_diamond_ranked_10min.csv")
DataHeaderDict = {'blueWardsPlaced' : FalseValue, #와드 설치
				'blueWardsDestroyed' : FalseValue, #와드 파괴
				'blueFirstBlood' : 1, #퍼스트 블러드
				'blueKills' : 0, #킬
  				'blueDeaths' : 9, #데스
  				'blueAssists' : 0, #어시
  				'blueEliteMonsters' : FalseValue, #버프
  				'blueDragons' : 0, #드래곤
  				'blueHeralds' : 0, #전령
  				'blueTowersDestroyed' : FalseValue, #타워 파괴
  				'blueTotalGold' : 0, #총합 골드
  				'blueAvgLevel' : 7.2, #평균 레벨
  				'blueTotalExperience' : FalseValue, #총합 레벨
  				'blueTotalMinionsKilled' : 0, #총 미니언 킬 수
  				'blueTotalJungleMinionsKilled' : FalseValue, #정글 몹
  				'blueGoldDiff' : -10000, #돈 차이
  				'blueExperienceDiff' : -0.5, #경험치 차이
  				'blueCSPerMin' : 20, #분 당 CS
  				'blueGoldPerMin' : 200, #분 당 골드
				}

Datas = []
for key in DataHeaderDict:
	if DataHeaderDict[key] != FalseValue:
		Datas.append(DataHeaderDict[key] / dataframe[key].max())
print(Datas)

ModelLoc = "E:\\GithubProjects\\KagglePractice\\LOL_Wins\\Model"
model = tf.keras.models.load_model(ModelLoc)

predict = model.predict([Datas])

for key in DataHeaderDict:
	if DataHeaderDict[key] != FalseValue:
		print(f"{key} : {DataHeaderDict[key]}")
print(f"승률 : {predict[0][0] * 100}%")