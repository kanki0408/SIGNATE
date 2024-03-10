#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Google Driveと接続を行います。これを行うことで、Driveにあるデータにアクセスできるようになります。
# 下記セルを実行すると、Googleアカウントのログインを求められますのでログインしてください。
from google.colab import drive
drive.mount('/content/drive')


# In[2]:


# 作業フォルダへの移動を行います。
# 人によって作業場所が異なるので、その場合作業場所を変更してください。
import os
os.chdir('/content/drive/MyDrive/コンペ/参加中コンペ') #ここを変更。


# In[3]:


import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submit.csv',index_col=0, header=None)
train.head()


# In[4]:


test.head()


# In[5]:


sample.head()


# In[11]:


train.describe()


# In[24]:


diabetes=train[train["Outcome"]==1]
no_diabetes=train[train["Outcome"]==0]


# In[27]:


len(diabetes)


# In[26]:


len(no_diabetes)


# In[45]:


import numpy as np
# 平均値と標準偏差を指定
mu = train['Pregnancies'].mean()
sigma = train['Pregnancies'].std()

# X軸の値を生成
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

# 正規分布の確率密度関数を計算
y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

# 正規分布をプロット
plt.plot(x, y)
plt.title('Normal Distribution (μ=0, σ=1)')
plt.xlabel('x')
plt.ylabel('Probability density')
plt.grid(True)
plt.show()


# In[12]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['Pregnancies'])
plt.xlabel('index')
plt.ylabel('Pregnancies')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[28]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['Pregnancies'])
plt.xlabel('index')
plt.ylabel('Pregnancies')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[29]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['Pregnancies'])
plt.xlabel('index')
plt.ylabel('Pregnancies')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[46]:


import numpy as np
# 平均値と標準偏差を指定
mu = train['Glucose'].mean()
sigma = train['Glucose'].std()

# X軸の値を生成
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

# 正規分布の確率密度関数を計算
y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

# 正規分布をプロット
plt.plot(x, y)
plt.title('Normal Distribution (μ=0, σ=1)')
plt.xlabel('x')
plt.ylabel('Probability density')
plt.grid(True)
plt.show()


# In[13]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['Glucose'])
plt.xlabel('index')
plt.ylabel('Glucose')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[30]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['Glucose'])
plt.xlabel('index')
plt.ylabel('Glucose')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[31]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['Glucose'])
plt.xlabel('index')
plt.ylabel('Glucose')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[15]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['BloodPressure'])
plt.xlabel('index')
plt.ylabel('BloodPressure')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[32]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['BloodPressure'])
plt.xlabel('index')
plt.ylabel('BloodPressure')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[33]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['BloodPressure'])
plt.xlabel('index')
plt.ylabel('BloodPressure')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[16]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['SkinThickness'])
plt.xlabel('index')
plt.ylabel('SkinThickness')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[35]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['SkinThickness'])
plt.xlabel('index')
plt.ylabel('SkinThickness')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[34]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['SkinThickness'])
plt.xlabel('index')
plt.ylabel('SkinThickness')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[17]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['Insulin'])
plt.xlabel('index')
plt.ylabel('Insulin')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[36]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['Insulin'])
plt.xlabel('index')
plt.ylabel('Insulin')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[37]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['Insulin'])
plt.xlabel('index')
plt.ylabel('Insulin')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[18]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['BMI'])
plt.xlabel('index')
plt.ylabel('BMI')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[38]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['BMI'])
plt.xlabel('index')
plt.ylabel('BMI')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[39]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['BMI'])
plt.xlabel('index')
plt.ylabel('BMI')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[19]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['Age'])
plt.xlabel('index')
plt.ylabel('Age')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[40]:


import matplotlib.pyplot as plt
plt.scatter(diabetes['index'], diabetes['Age'])
plt.xlabel('index')
plt.ylabel('Age')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[41]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['Age'])
plt.xlabel('index')
plt.ylabel('Age')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[20]:


import matplotlib.pyplot as plt
plt.scatter(train['index'], train['DiabetesPedigreeFunction'])
plt.xlabel('index')
plt.ylabel('DiabetesPedigreeFunction')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[42]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['DiabetesPedigreeFunction'])
plt.xlabel('index')
plt.ylabel('DiabetesPedigreeFunction')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[43]:


import matplotlib.pyplot as plt
plt.scatter(no_diabetes['index'], no_diabetes['DiabetesPedigreeFunction'])
plt.xlabel('index')
plt.ylabel('DiabetesPedigreeFunction')
plt.title('Scatter Plot')
plt.grid(True)  # グリッドを表示する
plt.show()


# In[21]:


train_x=train.drop(["Outcome"],axis=1)
train_y=train["Outcome"]
test_x = test.copy()


# In[22]:


train_x = train_x.drop(["index"],axis=1)
test_x = test_x.drop(["index"],axis=1)


# In[71]:


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


# In[71]:





# In[71]:





# In[72]:


pred_label


# In[73]:


sample[1] = pred_label
sample.to_csv("submit.csv", header=None)


# In[73]:




