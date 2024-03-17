#!/usr/bin/env python
# coding: utf-8

# In[607]:


# Google Driveと接続を行います。これを行うことで、Driveにあるデータにアクセスできるようになります。
# 下記セルを実行すると、Googleアカウントのログインを求められますのでログインしてください。
from google.colab import drive
drive.mount('/content/drive')


# In[608]:


# 作業フォルダへの移動を行います。
# 人によって作業場所が異なるので、その場合作業場所を変更してください。
import os
os.chdir('/content/drive/MyDrive/コンペ/参加中コンペ') #ここを変更。


# In[609]:


import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submit.csv',index_col=0, header=None)
train.head()


# In[610]:


test.head()


# In[611]:


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


# In[611]:





# In[612]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import fetch_california_housing
train.hist(bins=30,figsize=(12,12))
plt.show()


# In[613]:


import seaborn as sns
sns.set(style="darkgrid")
from sklearn.datasets import fetch_california_housing

#図のサイズを表している
plt.figure(figsize=(3,6))
#箱ひげ図を作成すると書いてある
sns.boxplot(y=train["Glucose"])
plt.title("Boxplot")
plt.show()


# In[614]:


def plot_boxplot_and_hist(data,variable):
  #2つのMatplotlib.Axesからなる図(ax_boxとax_hist)
  #plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.50, 0.85)}) は、高さの比率が (0.50, 0.85) である2つのサブプロットを持つFigureオブジェクトを作成します。一方のサブプロットは箱ひげ図（ax_box）であり、もう一方はヒストグラム（ax_hist）です。sharex=True は、x軸を共有することを意味します。つまり、両方のサブプロットが同じx軸を共有します。
  #f, (ax_box, ax_hist) = ... のようにして、戻り値を f という変数に代入しています。これにより、作成された図全体 (f) とそれぞれのサブプロット (ax_box と ax_hist) にアクセスできます。
  f,(ax_box,ax_hist)=plt.subplots(2,sharex=True,
  gridspec_kw={"height_ratios":(0.50,0.85)})

  #グラフをそれぞれの軸に割り当てる
  #ヒストグラムと箱ひげ図の2つを作成している。またax=ax_boxとすることによってどこに描画するのかを決めている。
  sns.boxplot(x=data[variable],ax=ax_box)
  sns.histplot(data=data,x=variable,ax=ax_hist)

  #箱ひげ図のx軸のラベルを削除する
  ax_box.set(xlabel="")
  plt.title(variable)
  plt.show()


# In[615]:


plot_boxplot_and_hist(train,"Glucose")


# In[616]:


def find_limits(df,variable,fold):
  IQR = df[variable].quantile(0.75)-df[variable].quantile(0.25)
  lower_limit = df[variable].quantile(0.25)-(IQR*fold)
  upper_limit = df[variable].quantile(0.75)+(IQR*fold)
  return lower_limit,upper_limit


# In[617]:


lower_limit,upper_limit = find_limits(train , "Glucose",2)
print(lower_limit,upper_limit)


# In[618]:


outliers = np.where((train["Glucose"]>upper_limit) | (train["Glucose"]<lower_limit),True,False,)
outliers.sum()


# In[619]:


from feature_engine.outliers import Winsorizer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# In[620]:


breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
y = breast_cancer.target


# In[621]:


X.head()


# In[622]:


y


# In[622]:





# In[623]:


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


# In[624]:


diagnostic_plots(train,"Pregnancies")


# In[625]:


train.head()


# In[626]:


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


# In[627]:


diagnostic_plots(train,"SkinThickness")


# In[628]:


diagnostic_plots(train,"BloodPressure")


# In[629]:


plt.hist(train['Glucose'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[630]:


plt.hist(train['BloodPressure'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[631]:


plt.hist(train['SkinThickness'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[632]:


plt.hist(train['Insulin'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[633]:


plt.hist(train['BMI'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[634]:


plt.hist(train['Age'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[635]:


plt.hist(train['DiabetesPedigreeFunction'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[636]:


sample.head()


# In[637]:


train.describe()


# In[638]:


train_x=train.drop(["Outcome"],axis=1)
train_y=train["Outcome"]
test_x = test.copy()


# In[639]:


#相関関係の確認
train.corrwith(train["Outcome"])


# In[640]:


train_x = train_x.drop(["index"],axis=1)
test_x = test_x.drop(["index"],axis=1)


# In[641]:


lower_limit,upper_limit = find_limits(train_x,"Glucose",2)
lower_limit,upper_limit


# In[642]:


o_train_x = train_x.copy()
o_test_x = test_x.copy()
o2_train_x = train_x.copy()
o2_test_x = test_x.copy()


# In[643]:


o_train_x["Glucose"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_test_x["Glucose"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_train_x["Glucose"].min(),o_train_x["Glucose"].max()


# In[644]:


lower_limit,upper_limit = find_limits(train_x,"BloodPressure",2)
lower_limit,upper_limit


# In[645]:


o_train_x["BloodPressure"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_test_x["BloodPressure"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_train_x["BloodPressure"].min(),o_train_x["BloodPressure"].max()


# In[646]:


"""capper = Winsorizer(variables=["Glucose","BloodPressure"],capping_method="gaussian",tail="both",fold=2,)
capper.fit(o_train_x)"""


# In[647]:


"""o_train_x = capper.transform(o_train_x)
o_test_x = capper.transform(o_test_x)"""


# In[648]:


diagnostic_plots(train,"BloodPressure")


# In[649]:


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


# In[650]:


train_x.head()


# In[651]:


st_train_x.head()


# In[652]:


st_test_x.head()


# In[652]:





# In[653]:


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


# In[654]:


diagnostic_plots(train,"SkinThickness")


# In[655]:


from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(lambda x: np.power(x,0.3))
train_x2["SkinThickness"] = transformer.transform(train_x["SkinThickness"])
test_x2["SkinThickness"] = transformer.transform(test_x["SkinThickness"])


# In[656]:


diagnostic_plots(train_x2,"SkinThickness")


# In[657]:


diagnostic_plots(train,"Insulin")


# In[658]:


from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(lambda x: np.power(x,0.5))
train_x2["Insulin"] = transformer.transform(train_x["Insulin"])
test_x2["Insulin"] = transformer.transform(test_x["Insulin"])
diagnostic_plots(train_x2,"Insulin")


# In[659]:


diagnostic_plots(train,"DiabetesPedigreeFunction")


# In[659]:





# In[660]:


train_x2.hist(bins=30,figsize=(12,12))
plt.show()


# In[661]:


diagnostic_plots(train_x,"SkinThickness")


# In[662]:


train_x.head()


# In[663]:


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
for tr_idx,va_idx in kf.split(o_train_x):
  #学習データを学習データとバリデーションデータに分ける
  tr_x,va_x=o_train_x.iloc[tr_idx],o_train_x.iloc[va_idx]
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
#accuary0.8010
#accuary0.8027
#logloss:0.4682


# In[664]:


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


# In[665]:


test_x.head()


# In[666]:


diagnostic_plots(train,"Pregnancies")


# In[666]:





# In[667]:


pip install feature-engine


# In[668]:


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


# In[669]:


from xgboost import XGBClassifier

model_xgb=XGBClassifier(n_estimators=20,random_state=71)
model_xgb.fit(o_train_x,train_y)
pred_xgb=model_xgb.predict_proba(test_x)[:,1]
model_lr=LogisticRegression(solver="lbfgs",max_iter=300)
model_lr.fit(train_x2,train_y)
pred_lr=model_lr.predict_proba(test_x2)[:,1]
pred=pred_xgb*0.8+pred_lr*0.2
pred_label=np.where(pred>0.5,1,0)


# In[670]:


pred


# In[671]:


pred_label


# In[672]:


sample[1] = pred_label
sample.to_csv("submit.csv", header=None)


# In[672]:





# In[672]:




