import tensorflow as tf

DataHeaderDict = {#'와드 설치' : 10,
				#'와드 파괴' : 0, 
				'퍼스트 블러드 여부 1/0' : 1, 
				'킬' : 0, 
  				'데스' : 0, 
  				#'어시' : 10, 
  				#'버프' : 5, 
  				'드래곤' : 0, 
  				#'전령' : 1, #전령
  				#'타워 파괴' : 1, 
  				'총합 골드' : 0, 
  				#'평균 레벨' : 7, 
  				#'총합 레벨' : 35, 
  				'총 미니언 킬 수' : 100, 
  				#'정글 몹' : 0, 
  				#'돈 차이' : 500, 
  				#'경험치 차이' : 120, 
  				#'분 당 CS' : 7.8, 
  				#'분 당 골드' : 170
				}

Datas = []
for key in DataHeaderDict:
	Datas.append(DataHeaderDict[key])
print(Datas)

ModelLoc = "E:\\GithubProjects\\KagglePractice\\LOL_Wins\\Model"
model = tf.keras.models.load_model(ModelLoc)

predict = model.predict([Datas])

for key in DataHeaderDict:
	print(f"{key} : {DataHeaderDict[key]}")
print(f"승률 : {predict[0][0] * 100}%")