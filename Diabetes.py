#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Google Driveと接続を行います。これを行うことで、Driveにあるデータにアクセスできるようになります。
# 下記セルを実行すると、Googleアカウントのログインを求められますのでログインしてください。
from google.colab import drive
drive.mount('/content/drive')


# In[3]:


# 作業フォルダへの移動を行います。
# 人によって作業場所が異なるので、その場合作業場所を変更してください。
import os
os.chdir('/content/drive/MyDrive/コンペ/参加中コンペ') #ここを変更。


# In[17]:


pip install feature-engine


# In[18]:


pip install feature-engine


# In[2313]:


import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submit.csv',index_col=0, header=None)
train.head()


# In[2314]:


test.head()


# In[2315]:


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


# In[2315]:





# In[2316]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import fetch_california_housing
train.hist(bins=30,figsize=(12,12))
plt.show()


# In[2383]:


import seaborn as sns
sns.set(style="darkgrid")
from sklearn.datasets import fetch_california_housing

#図のサイズを表している
plt.figure(figsize=(3,6))
#箱ひげ図を作成すると書いてある
sns.boxplot(y=train["Glucose"])
plt.title("Boxplot")
plt.show()


# In[2384]:


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


# In[2385]:


plot_boxplot_and_hist(train,"Glucose")


# In[2386]:


def find_limits(df,variable,fold):
  IQR = df[variable].quantile(0.75)-df[variable].quantile(0.25)
  lower_limit = df[variable].quantile(0.25)-(IQR*fold)
  upper_limit = df[variable].quantile(0.75)+(IQR*fold)
  return lower_limit,upper_limit


# In[2387]:


lower_limit,upper_limit = find_limits(train , "Glucose",2)
print(lower_limit,upper_limit)


# In[2388]:


outliers = np.where((train["Glucose"]>upper_limit) | (train["Glucose"]<lower_limit),True,False,)
outliers.sum()


# In[2389]:


from feature_engine.outliers import Winsorizer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# In[2390]:


breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
y = breast_cancer.target


# In[2391]:


X.head()


# In[2392]:


y


# In[2392]:





# In[2393]:


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


# In[2394]:


diagnostic_plots(train,"Pregnancies")


# In[2395]:


train.head()


# In[2396]:


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


# In[2397]:


diagnostic_plots(train,"SkinThickness")


# In[2398]:


diagnostic_plots(train,"BloodPressure")


# In[2399]:


plt.hist(train['Glucose'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[2400]:


plt.hist(train['BloodPressure'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[2401]:


plt.hist(train['SkinThickness'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[2402]:


plt.hist(train['Insulin'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[2403]:


plt.hist(train['BMI'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[2404]:


plt.hist(train['Age'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[2405]:


plt.hist(train['DiabetesPedigreeFunction'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[2406]:


sample.head()


# In[2407]:


train.describe()


# In[2408]:


train_x=train.drop(["Outcome"],axis=1)
train_y=train["Outcome"]
test_x = test.copy()


# In[2409]:


#相関関係の確認
train.corrwith(train["Outcome"])


# In[2410]:


train_x = train_x.drop(["index"],axis=1)
test_x = test_x.drop(["index"],axis=1)


# In[2411]:


lower_limit,upper_limit = find_limits(train_x,"Glucose",2)
lower_limit,upper_limit


# In[2412]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
t_train_x = poly.fit_transform(train_x)
t_test_x = poly.fit_transform(test_x)
t_train_x


# In[2413]:


poly.get_feature_names_out()


# In[2414]:


t_train_x = pd.DataFrame(t_train_x,columns=poly.get_feature_names_out())
t_test_x = pd.DataFrame(t_test_x,columns=poly.get_feature_names_out())


# In[2415]:


check = t_train_x
t_train_x["Outcome"] = train["Outcome"]


# In[2416]:


t_train_x.head()


# In[2417]:


t_train_x.corrwith(t_train_x["Outcome"])


# In[2418]:


high_correlation_columns = []
corr_series = t_train_x.corrwith(t_train_x["Outcome"])
#corr_series.items()によってカラム名と値を取り出している
for column, correlation in corr_series.items():
    if abs(correlation) > 0.2:  # 絶対値が0.2よりも大きい場合
        high_correlation_columns.append(column)
OK = [ 'Pregnancies Glucose',
 'Pregnancies BloodPressure',
 'Pregnancies BMI',
 'Pregnancies Age',
 'Glucose BMI',
 'Glucose Age',
 'BloodPressure BMI',
 'BloodPressure Age',
 'BMI Age']
OK


# In[2419]:


t_train_x[OK].head()


# In[2420]:


ok_train_x.head()


# In[2421]:


o_train_x = train_x.copy()
o_test_x = test_x.copy()
o2_train_x = train_x.copy()
o2_test_x = test_x.copy()


# In[2422]:


o_train_x["Glucose"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_test_x["Glucose"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_train_x["Glucose"].min(),o_train_x["Glucose"].max()


# In[2423]:


lower_limit,upper_limit = find_limits(train_x,"BloodPressure",2)
lower_limit,upper_limit


# In[2424]:


o_train_x["BloodPressure"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_test_x["BloodPressure"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_train_x["BloodPressure"].min(),o_train_x["BloodPressure"].max()


# In[2425]:


lower_limit,upper_limit = find_limits(train_x,"DiabetesPedigreeFunction",2)
lower_limit,upper_limit


# In[2426]:


o_train_x["DiabetesPedigreeFunction"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_test_x["DiabetesPedigreeFunction"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_train_x["DiabetesPedigreeFunction"].min(),o_train_x["DiabetesPedigreeFunction"].max()


# In[2427]:


diagnostic_plots(train,"BloodPressure")


# In[2428]:


OK


# In[2429]:


ok_train_x = o_train_x.copy()
ok_test_x = o_test_x.copy()
ok_train_x['Pregnancies BMI'] = t_train_x['Pregnancies BMI']
ok_test_x['Pregnancies BMI'] = t_test_x['Pregnancies BMI']


# In[2430]:


lower_limit,upper_limit = find_limits(ok_train_x,"Pregnancies BMI",2)
lower_limit,upper_limit
ok_train_x["Pregnancies BMI"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
ok_test_x["Pregnancies BMI"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
ok_train_x["Pregnancies BMI"].min(),ok_train_x["Pregnancies BMI"].max()


# In[2430]:





# In[2431]:


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


# In[2432]:


train_x.head()


# In[2433]:


st_train_x.head()


# In[2434]:


st_test_x.head()


# In[2434]:





# In[2435]:


test_x2=test_x.copy()
train_x2=train_x.copy()
train_x2["Pregnancies BMI"] = ok_train_x["Pregnancies BMI"]
test_x2["Pregnancies BMI"] = ok_test_x["Pregnancies BMI"]
# 変換後のデータで各列を置換
train_x2['SkinThickness'] = np.log1p(train_x2['SkinThickness'])
train_x2['Insulin'] = np.log1p(train_x2['Insulin'])
train_x2['Age'] = np.log1p(train_x2['Age'])
train_x2['DiabetesPedigreeFunction'] = np.log1p(train_x2['DiabetesPedigreeFunction'])
train_x2['Pregnancies BMI'] = np.log1p(train_x2['Pregnancies BMI'])
test_x2['SkinThickness'] = np.log1p(test_x2['SkinThickness'])
test_x2['Insulin'] = np.log1p(test_x2['Insulin'])
test_x2['Age'] = np.log1p(test_x2['Age'])
test_x2['DiabetesPedigreeFunction'] = np.log1p(test_x2['DiabetesPedigreeFunction'])
test_x2['Pregnancies BMI'] = np.log1p(test_x2['Pregnancies BMI'])


# In[2436]:


diagnostic_plots(train,"SkinThickness")


# In[2437]:


diagnostic_plots(ok_train_x,"Pregnancies BMI")


# In[2438]:


diagnostic_plots(train_x2,"SkinThickness")


# In[2439]:


diagnostic_plots(train,"Insulin")


# In[2439]:





# In[2440]:


diagnostic_plots(train,"DiabetesPedigreeFunction")


# In[2440]:





# In[2441]:


train_x2.hist(bins=30,figsize=(12,12))
plt.show()


# In[2442]:


diagnostic_plots(train_x,"SkinThickness")


# In[2443]:


train_x.head()


# In[2444]:


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
for tr_idx,va_idx in kf.split(ok_train_x):
  #学習データを学習データとバリデーションデータに分ける
  tr_x,va_x=ok_train_x.iloc[tr_idx],ok_train_x.iloc[va_idx]
  tr_y,va_y=train_y.iloc[tr_idx],train_y.iloc[va_idx]
  #特徴量と目的変数をxgboostのデータ構造に変換する
  dtrain = xgb.DMatrix(tr_x,label=tr_y)
  dvalid = xgb.DMatrix(va_x, label = va_y)
  dtest = xgb.DMatrix(ok_test_x)
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
"""logloss:0.4546
accuracy:0.8033"""


# In[2445]:


test_x.head()


# In[2446]:


diagnostic_plots(train,"Pregnancies")


# In[2446]:





# In[2447]:


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
#logloss:0.4807
#accuracy:0.7770


# In[2448]:


from xgboost import XGBClassifier

model_xgb=XGBClassifier(n_estimators=20,random_state=71)
model_xgb.fit(o_train_x,train_y)
pred_xgb=model_xgb.predict_proba(o_test_x)[:,1]
model_lr=LogisticRegression(solver="lbfgs",max_iter=300)
model_lr.fit(train_x2,train_y)
pred_lr=model_lr.predict_proba(test_x2)[:,1]
pred=pred_xgb*0.8+pred_lr*0.2
pred_label=np.where(pred>0.5,1,0)


# In[2449]:


pred


# In[2450]:


pred_label


# In[2451]:


sample[1] = pred_label
sample.to_csv("submit.csv", header=None)


# In[2451]:





# In[2451]:





# In[2451]:




