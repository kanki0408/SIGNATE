#!/usr/bin/env python
# coding: utf-8

# In[103]:


# Google Driveと接続を行います。これを行うことで、Driveにあるデータにアクセスできるようになります。
# 下記セルを実行すると、Googleアカウントのログインを求められますのでログインしてください。
from google.colab import drive
drive.mount('/content/drive')


# In[104]:


# 作業フォルダへの移動を行います。
# 人によって作業場所が異なるので、その場合作業場所を変更してください。
import os
os.chdir('/content/drive/MyDrive/コンペ/参加中コンペ') #ここを変更。


# In[105]:


import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submit.csv',index_col=0, header=None)
train.head()


# In[106]:


test.head()


# In[107]:


sample.head()


# In[108]:


train.describe()


# In[131]:


train_x=train.drop(["Outcome"],axis=1)
train_y=train["Outcome"]
test_x = test.copy()


# In[110]:


diabetes=train[train["Outcome"]==1]
no_diabetes=train[train["Outcome"]==0]


# In[111]:


len(diabetes)


# In[112]:


len(no_diabetes)


# In[112]:





# In[113]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['Pregnancies'])
plt.xlabel('index')
plt.ylabel('Pregnancies')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['Pregnancies'])
plt.xlabel('index')
plt.ylabel('Pregnancies')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['Pregnancies'])
plt.xlabel('index')
plt.ylabel('Pregnancies')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(diabetes['Pregnancies'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


plt.hist(no_diabetes['Pregnancies'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


print(train_x.columns)


# In[114]:


from sklearn.preprocessing import StandardScaler
st_train_x = train_x.copy()
# 学習データに基づいて複数列の標準化を定義
scaler = StandardScaler()
scaler.fit(train_x)
scaler.fit(test_x)
# 変換後のデータで各列を置換
st_train_x = scaler.transform(train_x)
st_test_x = scaler.transform(test_x)

st_train_x=pd.DataFrame(st_train_x, columns=train_x.columns, index=train_x.index)
st_test_x=pd.DataFrame(st_test_x, columns=test_x.columns, index=test_x.index)


# In[ ]:


train_y.head()


# In[115]:


from sklearn.decomposition import PCA

# データは標準化などのスケールを揃える前処理が行われているものとする

# 学習データに基づいてPCAによる変換を定義
pca = PCA(n_components=2)
pca.fit(st_train_x)

# 変換の適用
pca_train_x = pca.transform(st_train_x)
pca_df = pd.DataFrame(pca_train_x)
pca_df["Outcome"] = train_y
pca_df.head()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
for i in pca_df["Outcome"].unique():
    tmp = pca_df.loc[pca_df["Outcome"]==i]
    plt.scatter(tmp[0], tmp[1])


# In[ ]:


st_train_x.head()


# In[117]:


get_ipython().system('pip install bhtsne')
import bhtsne


# In[119]:


# データは標準化などのスケールを揃える前処理が行われているものとする

# t-sneによる変換
embedded = bhtsne.tsne(st_train_x, dimensions=2, rand_seed=71)


# In[142]:


embedded2 = bhtsne.tsne(st_test_x, dimensions=2, rand_seed=71)


# In[125]:





# In[124]:


len(embedded)


# In[123]:


plt.scatter(embedded[:,0],embedded[:,1])


# In[143]:


tsne_df = pd.DataFrame(embedded)
tsne_df["outcome"] = train_y


# In[144]:


tsne_train_x =pd.DataFrame(embedded)
tsne_test_x =pd.DataFrame(embedded2)


# In[ ]:





# In[128]:


tsne_df.head()


# In[139]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# データは標準化などのスケールを揃える前処理が行われているものとする

# 学習データに基づいてLDAによる変換を定義
lda = LDA(n_components=1)
lda.fit(st_train_x, train_y)

# 変換の適用
lda_train_x = lda.transform(st_train_x)
lda_test_x = lda.transform(st_test_x)
lda_df = pd.DataFrame(lda_train_x)


# In[138]:


train_x.head()


# In[129]:


for i in tsne_df["outcome"].unique():
    tmp = tsne_df.loc[tsne_df["outcome"]==i]
    plt.scatter(tmp[0], tmp[1])


# In[ ]:



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
for i in pca_df["Outcome"].unique():
    tmp = pca_df.loc[pca_df["Outcome"]==i]
    plt.scatter(tmp[0], tmp[1])


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['Glucose'])
plt.xlabel('index')
plt.ylabel('Glucose')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['Glucose'])
plt.xlabel('index')
plt.ylabel('Glucose')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(diabetes['Glucose'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


plt.hist(no_diabetes['Glucose'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['Glucose'])
plt.xlabel('index')
plt.ylabel('Glucose')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['BloodPressure'])
plt.xlabel('index')
plt.ylabel('BloodPressure')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['BloodPressure'])
plt.xlabel('index')
plt.ylabel('BloodPressure')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(diabetes['BloodPressure'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['BloodPressure'])
plt.xlabel('index')
plt.ylabel('BloodPressure')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(no_diabetes['BloodPressure'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['SkinThickness'])
plt.xlabel('index')
plt.ylabel('SkinThickness')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['SkinThickness'])
plt.xlabel('index')
plt.ylabel('SkinThickness')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(diabetes['SkinThickness'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['SkinThickness'])
plt.xlabel('index')
plt.ylabel('SkinThickness')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(no_diabetes['SkinThickness'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['Insulin'])
plt.xlabel('index')
plt.ylabel('Insulin')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['Insulin'])
plt.xlabel('index')
plt.ylabel('Insulin')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(diabetes['Insulin'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['Insulin'])
plt.xlabel('index')
plt.ylabel('Insulin')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(no_diabetes['Insulin'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['BMI'])
plt.xlabel('index')
plt.ylabel('BMI')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['BMI'])
plt.xlabel('index')
plt.ylabel('BMI')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(diabetes['BMI'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['BMI'])
plt.xlabel('index')
plt.ylabel('BMI')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(no_diabetes['BMI'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['Age'])
plt.xlabel('index')
plt.ylabel('Age')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['Age'])
plt.xlabel('index')
plt.ylabel('Age')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(diabetes['Age'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['Age'])
plt.xlabel('index')
plt.ylabel('Age')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(no_diabetes['Age'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['DiabetesPedigreeFunction'])
plt.xlabel('index')
plt.ylabel('DiabetesPedigreeFunction')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['DiabetesPedigreeFunction'])
plt.xlabel('index')
plt.ylabel('DiabetesPedigreeFunction')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(diabetes['DiabetesPedigreeFunction'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['DiabetesPedigreeFunction'])
plt.xlabel('index')
plt.ylabel('DiabetesPedigreeFunction')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[ ]:


plt.hist(no_diabetes['DiabetesPedigreeFunction'], color='skyblue', edgecolor='black')
# グラフのタイトルとラベルの設定
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# グリッド線の表示
plt.grid(True)

# グラフの表示
plt.show()


# In[ ]:


diabetes.head()


# In[ ]:


#相関関係の確認
train.corrwith(train["Outcome"])


# In[ ]:





# In[ ]:


train_x=train.drop(["Outcome"],axis=1)
train_y=train["Outcome"]
test_x = test.copy()


# In[ ]:


print(train.columns)


# In[ ]:


train_x = train_x.drop(["index"],axis=1)
test_x = test_x.drop(["index"],axis=1)


# In[ ]:


train_x.head()


# In[ ]:





# In[145]:


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
for tr_idx,va_idx in kf.split(lda_df):
  #学習データを学習データとバリデーションデータに分ける
  tr_x,va_x=tsne_train_x.iloc[tr_idx],tsne_train_x.iloc[va_idx]
  tr_y,va_y=train_y.iloc[tr_idx],train_y.iloc[va_idx]
  #特徴量と目的変数をxgboostのデータ構造に変換する
  dtrain = xgb.DMatrix(tr_x,label=tr_y)
  dvalid = xgb.DMatrix(va_x, label = va_y)
  dtest = xgb.DMatrix(tsne_test_x)
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


# In[ ]:





# In[ ]:





# In[ ]:


pred_label


# In[ ]:


sample[1] = pred_label
sample.to_csv("submit.csv", header=None)


# In[ ]:




