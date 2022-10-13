import tensorflow as tf
import pandas as pd

dataframe = pd.read_csv("E:\\GithubProjects\\KagglePractice\\University\\gpascore.csv")
ModelLoc = "E:\\GithubProjects\\KagglePractice\\University\\Model"

model = tf.keras.models.load_model(ModelLoc)

predict = model.predict([ [750, 3.70, 3], [400, 2.2, 1] ])
print(predict)