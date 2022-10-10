# https://www.kaggle.com/competitions/titanic/data?select=test.csv
# https://kaggle-kr.tistory.com/17?category=868316

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use('seaborn')
sns.set(font_scale = 2.5)

import missingno as msno

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv("E:\\GithubProjects\\KagglePractice\\Titanic\\train.csv")
df_test = pd.read_csv("E:\\GithubProjects\\KagglePractice\\Titanic\\test.csv")

# ======== 잘 가져와지는 지 보기 =========
# print(df_train.head())
# print(df_train.describe())


# ======== Null 구하기 ========
# 1. 반복
# for col in df_train.columns:
# 	msg = 'column: {:>10}\t Percent of Nan Value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
# 	print(msg)

# 2. 함수
# print(df_train.agg(lambda x: sum(x.isnull()) / x.shape[0]))

# Age, Cabin, Embarked 에 Null 존재


# ======== 생존율 확인 ========
# f, ax = plt.subplots(1, 2, figsize=(18, 8))

# df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
# ax[0].set_title('Pie plot - Survived')
# ax[0].set_ylabel('')

# plt.show()


# ======== 데이터 분석 ========
# Data = df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
# print(Data)
# 클래스 별 총합

# Data = df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
# print(Data)
# 클래스 별 생존자

# print(pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True))
# #.style.background_gradient(cmap='summer_r'))
# 스타일은 vsc 에서 안 보이는 듯함

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by="Survived", ascending=False).plot.bar()