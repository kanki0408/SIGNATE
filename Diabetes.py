#!/usr/bin/env python
# coding: utf-8

# In[490]:


# Google Driveと接続を行います。これを行うことで、Driveにあるデータにアクセスできるようになります。
# 下記セルを実行すると、Googleアカウントのログインを求められますのでログインしてください。
from google.colab import drive
drive.mount('/content/drive')


# In[491]:


# 作業フォルダへの移動を行います。
# 人によって作業場所が異なるので、その場合作業場所を変更してください。
import os
os.chdir('/content/drive/MyDrive/コンペ/参加中コンペ') #ここを変更。


# In[492]:


import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submit.csv',index_col=0, header=None)
train.head()


# In[493]:


test.head()


# In[494]:


import matplotlib.pyplot as plt
plt.hist(train['Pregnancies'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[495]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import fetch_california_housing
train.hist(bins=30,figsize=(12,12))
plt.show()


# In[496]:


def diagnostic_plots(df,variable):
  #図のサイズ
  plt.figure(figsize=(15,6))

  #作成された図をどの位置に表示するのかということを意味する。
  #1行2列のグリッドにおいて、1行目の左側（つまり1番目の位置）にサブプロットを配置することを意味します。
  plt.subplot(1,2,1)
  df[variable].value_counts().sort_index().plot.bar()
  plt.title(f"Hitogram of {variable}")

  plt.subplot(1,2,2)
  #variableというデータ実際の図と、正規分布の場合の図をプロットしている
  stats.probplot(df[variable],dist="norm",plot=plt)
  plt.title(f"Q-Q plot of {variable}")

  plt.show


# In[497]:


diagnostic_plots(train,"Pregnancies")


# In[498]:


train.head()


# In[499]:


def diagnostic_plots(df,variable):
  #図のサイズ
  plt.figure(figsize=(15,6))

  #作成された図をどの位置に表示するのかということを意味する。
  #1行2列のグリッドにおいて、1行目の左側（つまり1番目の位置）にサブプロットを配置することを意味します。
  plt.subplot(1,2,1)
  df[variable].hist(bins=30)
  plt.title(f"Hitogram of {variable}")

  plt.subplot(1,2,2)
  #variableというデータ実際の図と、正規分布の場合の図をプロットしている
  stats.probplot(df[variable],dist="norm",plot=plt)
  plt.title(f"Q-Q plot of {variable}")

  plt.show


# In[500]:


diagnostic_plots(train,"SkinThickness")


# In[501]:


diagnostic_plots(train,"BloodPressure")


# In[502]:


plt.hist(train['Glucose'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[503]:


plt.hist(train['BloodPressure'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[504]:


plt.hist(train['SkinThickness'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[505]:


plt.hist(train['Insulin'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[506]:


plt.hist(train['BMI'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[507]:


plt.hist(train['Age'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[508]:


plt.hist(train['DiabetesPedigreeFunction'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[509]:


sample.head()


# In[510]:


train.describe()


# In[511]:


train_x=train.drop(["Outcome"],axis=1)
train_y=train["Outcome"]
test_x = test.copy()


# In[512]:


#相関関係の確認
train.corrwith(train["Outcome"])


# In[513]:


train_x = train_x.drop(["index"],axis=1)
test_x = test_x.drop(["index"],axis=1)


# In[514]:


diagnostic_plots(train,"BloodPressure")


# In[515]:


from sklearn.preprocessing import StandardScaler
# 学習データに基づいて複数列の標準化を定義
scaler = StandardScaler()
scaler.fit(train_x)
scaler.fit(test_x)
# 変換後のデータで各列を置換
st_train_x = scaler.transform(train_x)
st_test_x = scaler.transform(test_x)

st_train_x=pd.DataFrame(st_train_x, columns=train_x.columns, index=train_x.index)
st_test_x=pd.DataFrame(st_test_x, columns=test_x.columns, index=test_x.index)


# In[516]:


train_x.head()


# In[517]:


st_train_x.head()


# In[518]:


st_test_x.head()


# In[519]:


from sklearn.model_selection import KFold
from sklearn.metrics import log_loss , accuracy_score
import xgboost as xgb
from sklearn.metrics import log_loss
import numpy as np
from sklearn.metrics import log_loss , accuracy_score

scores_accuracy = []
scores_logloss =[]
#クロスバリデーションを行う
#学習データを4分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
kf=KFold(n_splits=4 , shuffle=True , random_state = 71)
for tr_idx,va_idx in kf.split(train_x):
  #学習データを学習データとバリデーションデータに分ける
  tr_x,va_x=train_x.iloc[tr_idx],train_x.iloc[va_idx]
  tr_y,va_y=train_y.iloc[tr_idx],train_y.iloc[va_idx]
  #特徴量と目的変数をxgboostのデータ構造に変換する
  dtrain = xgb.DMatrix(tr_x,label=tr_y)
  dvalid = xgb.DMatrix(va_x, label = va_y)
  dtest = xgb.DMatrix(test_x)
  #ハイパーパラメータの設定
  #silent:1によってが学習中のメッセージを抑制するようになっている
  #random_stateをせっていすることによって再現性を保つことが出来るようにしている。
  params = {"objective": "binary:logistic","silent":1,"random_state":71}
  num_round = 50;

  watchlist = [(dtrain,"train"),(dvalid,"eval")]
  model = xgb.train(params,dtrain,num_round,evals=watchlist)

  va_pred = model.predict(dvalid)
  #loglossはロジスティック損失を表しており、ロジスティック損失は、確率予測の正確さを測るための指標のひとつで、誤差が大きいほど損失が指数関数的に大きくなる特徴があります。
  score = log_loss(va_y,va_pred)
  accuracy = accuracy_score(va_y,va_pred>0.5)
  scores_logloss.append(score)
  scores_accuracy.append(accuracy)

print(f"logloss:{np.mean(scores_logloss):.4f}")
print(f"accuracy:{np.mean(scores_accuracy):.4f}")

pred = model.predict(dtest)
pred_label=np.where(pred>0.5,1,0)


# In[520]:


test_x2=test_x.copy()
train_x2=train_x.copy()
# 変換後のデータで各列を置換
train_x2['SkinThickness'] = np.log1p(train_x2['SkinThickness'])
train_x2['Insulin'] = np.log1p(train_x2['Insulin'])
train_x2['Age'] = np.log1p(train_x2['Age'])
train_x2['DiabetesPedigreeFunction'] = np.log1p(train_x2['DiabetesPedigreeFunction'])
test_x2['SkinThickness'] = np.log1p(test_x2['SkinThickness'])
test_x2['Insulin'] = np.log1p(test_x2['Insulin'])
test_x2['Age'] = np.log1p(test_x2['Age'])
test_x2['DiabetesPedigreeFunction'] = np.log1p(test_x2['DiabetesPedigreeFunction'])


# In[521]:


diagnostic_plots(train,"SkinThickness")


# In[522]:


from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(lambda x: np.power(x,0.3))
train_x2["SkinThickness"] = transformer.transform(train_x["SkinThickness"])
test_x2["SkinThickness"] = transformer.transform(test_x["SkinThickness"])


# In[523]:


diagnostic_plots(train_x2,"SkinThickness")


# In[524]:


diagnostic_plots(train,"Insulin")


# In[525]:


from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(lambda x: np.power(x,0.5))
train_x2["Insulin"] = transformer.transform(train_x["Insulin"])
test_x2["Insulin"] = transformer.transform(test_x["Insulin"])
diagnostic_plots(train_x2,"Insulin")


# In[526]:


diagnostic_plots(train,"DiabetesPedigreeFunction")


# In[526]:





# In[527]:


train_x2.hist(bins=30,figsize=(12,12))
plt.show()


# In[528]:


diagnostic_plots(train_x,"SkinThickness")


# In[529]:


train_x.head()


# In[530]:


from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

scores_accuracy = []
scores_logloss =[]
#クロスバリデーションを行う
#学習データを4分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
kf=KFold(n_splits=4 , shuffle=True , random_state = 71)
for tr_idx,va_idx in kf.split(train_x2):
  #学習データを学習データとバリデーションデータに分ける
  tr_x,va_x=train_x2.iloc[tr_idx],train_x2.iloc[va_idx]
  tr_y,va_y=train_y.iloc[tr_idx],train_y.iloc[va_idx]

  # ニューラルネットモデルの構築
  model = Sequential()
  model.add(Dense(256, activation='relu', input_shape=(train_x2.shape[1],)))
  model.add(Dropout(0.2))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy',
               optimizer='adam', metrics=['accuracy'])

  # 学習の実行
  # バリデーションデータもモデルに渡し、学習の進行とともにスコアがどう変わるかモニタリングする
  batch_size = 128
  epochs = 10
  history = model.fit(tr_x, tr_y,
                      batch_size=batch_size, epochs=epochs,
                      verbose=1, validation_data=(va_x, va_y))

  va_pred = model.predict(va_x)
  #loglossはロジスティック損失を表しており、ロジスティック損失は、確率予測の正確さを測るための指標のひとつで、誤差が大きいほど損失が指数関数的に大きくなる特徴があります。
  score = log_loss(va_y,va_pred)
  accuracy = accuracy_score(va_y,va_pred>0.5)
  scores_logloss.append(score)
  scores_accuracy.append(accuracy)

print(f"logloss:{np.mean(scores_logloss):.4f}")
print(f"accuracy:{np.mean(scores_accuracy):.4f}")

pred = model.predict(test_x)
pred_label=np.where(pred>0.5,1,0)
#logloss:0.4759
#accuracy:0.7827


# In[531]:


test_x.head()


# In[532]:


diagnostic_plots(train,"Pregnancies")


# In[532]:





# In[533]:


pip install feature-engine


# In[534]:


from sklearn.linear_model import LogisticRegression
scores_accuracy = []
scores_logloss =[]
#クロスバリデーションを行う
#学習データを4分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
kf=KFold(n_splits=4 , shuffle=True , random_state = 71)
for tr_idx,va_idx in kf.split(train_x2):
  #学習データを学習データとバリデーションデータに分ける
  tr_x,va_x=train_x2.iloc[tr_idx],train_x2.iloc[va_idx]
  tr_y,va_y=train_y.iloc[tr_idx],train_y.iloc[va_idx]
  model_lr=LogisticRegression(solver="lbfgs",max_iter=300)
  model_lr.fit(tr_x,tr_y)
  va_pred=model_lr.predict_proba(va_x)[:,1]

  #loglossはロジスティック損失を表しており、ロジスティック損失は、確率予測の正確さを測るための指標のひとつで、誤差が大きいほど損失が指数関数的に大きくなる特徴があります。
  score = log_loss(va_y,va_pred)
  accuracy = accuracy_score(va_y,va_pred>0.5)
  scores_logloss.append(score)
  scores_accuracy.append(accuracy)

print(f"logloss:{np.mean(scores_logloss):.4f}")
print(f"accuracy:{np.mean(scores_accuracy):.4f}")
#logloss:0.4870
#accuracy:0.7710
#logloss:0.4841
#accuracy:0.7717
#logloss:0.4833
#accuracy:0.7727
#logloss:0.4826
#accuracy:0.7730
#logloss:0.4807
#accuracy:0.7743
#logloss:0.4805
#accuracy:0.7743
#logloss:0.4802
#accuracy:0.7743
#logloss:0.4805
#accuracy:0.7747
#logloss:0.4808
#accuracy:0.7757


# In[535]:


from xgboost import XGBClassifier

model_xgb=XGBClassifier(n_estimators=20,random_state=71)
model_xgb.fit(train_x,train_y)
pred_xgb=model_xgb.predict_proba(test_x)[:,1]
model_lr=LogisticRegression(solver="lbfgs",max_iter=300)
model_lr.fit(train_x2,train_y)
pred_lr=model_lr.predict_proba(test_x2)[:,1]
pred=pred_xgb*0.8+pred_lr*0.2
pred_label=np.where(pred>0.5,1,0)


# In[536]:


pred


# In[537]:


pred_label


# In[538]:


sample[1] = pred_label
sample.to_csv("submit.csv", header=None)


# In[538]:




