#!/usr/bin/env python
# coding: utf-8

# 線形回帰

# 線形回帰スクラッチ

# In[1]:


# PandasのDataFrame型のtrain_dataに格納
import pandas as pd
import numpy as np

train_data=pd.read_csv("/Users/chidayasuhiro/diveintocode-ml/train.csv")


# In[2]:


# Pandasからndarrayへ変更

X =train_data.loc[:,['GrLivArea','YearBuilt']]
y =train_data.loc[:,"SalePrice"]
Xandy=pd.concat([X,y],axis=1)

X_np=np.array(X)
y_np=np.array(y)
X_np=X_np.astype(float)


# In[3]:


# 分割
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X_np,y_np)

train_y=train_y.reshape(len(train_y),1)
test_y=test_y.reshape(len(test_y),1)


# In[17]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_X)
train_X1sd=scaler.transform(train_X)
test_X1sd=scaler.transform(test_X)


# 完成した雛形

# In[4]:


class ScratchLinearRegression():
    """
    線形回帰のスクラッチ実装
    
    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    no_bias : bool
      バイアス項を入れない場合はTrue
    verbose : bool
      学習過程を出力する場合はTrue
    
    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      訓練データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証データに対する損失の記録
    """
    
    def __init__(self, num_iter, lr, no_bias, verbose):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        # 学習過程を記録する配列を用意
        self.verbose_list=[]
        self.verbose_list_val=[]                     
                
    def _gradient_descent(self,X,y,i,X_val,y_val):
        
        a1=X@self.param-y
        a2=X.T@a1
        self.lr=np.array(self.lr)
        a3=self.param-(np.array(self.lr)*a2)/np.array(len(X))
        self.param=a3
        self.verbose_list.append(self.param)
        
        y_pred=X@a3
        e_sum=0
        for j in range(len(y_pred)):
            e_sum += (y_pred[j]-y[j])**2
        loss=e_sum/np.array(2*len(y_pred))
        self.loss[i]=loss
        self.coef=self.param
        
        if X_val is not None:
            y_pred_val=X_val@a3
            e_sum=0
            for j in range(len(y_pred_val)):
                e_sum += (y_pred_val[j]-y_val[j])**2
            loss=e_sum/np.array(2*len(y_pred_val))
            self.val_loss[i]=loss
                    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        線形回帰を学習する。検証データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            訓練データの特徴量
        y : 次の形のndarray, shape (n_samples,1)
            訓練データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証データの特徴量
        y_val : 次の形のndarray, shape (n_samples,1)
            検証データの正解値
        """
        #　定数項を作成
        if self.no_bias==False:
            X_bias=np.ones(len(X))                       
            X=np.insert(X,0,X_bias,axis=1)
            if X_val is not None:
                X_bias_val=np.ones(len(X_val))                       
                X_val=np.insert(X_val,0,X_bias_val,axis=1)
                                    
        #　初期値　０で設定
        param=np.zeros(X.shape[1])
        self.param=param.reshape(len(param),1)
        
        for i in range(self.iter):
            self._gradient_descent(X,y,i,X_val, y_val)
            
        #verboseをTrueにした際は学習過程を出力   
        if self.verbose:
            print(self.verbose_list)            
                               
    def predict(self, X):
        """
        線形回帰を使い推定する。
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル
        Returns
        -------
        y_pred  :  次の形のndarray, shape (n_samples,)
            線形回帰による推定結果
        """
        if not(X.shape[1]==len(self.coef)):            
            X_bias=np.ones(len(X))
            X1=np.insert(X,0,X_bias,axis=1)
        y_pred=X1@self.coef            
        return y_pred


# In[28]:


# 実装例
from sklearn.metrics import mean_absolute_error
SLR1=ScratchLinearRegression(num_iter=2000,lr=0.000000001,no_bias=False,verbose=False)
SLR1.fit(train_X,train_y,test_X,test_y)


# In[30]:


SLR1.coef


# In[89]:


SLR1.loss.astype(int)


# In[90]:


SLR1.val_loss.astype(int)


# In[29]:


import matplotlib.pyplot as plt
plt.plot(np.arange(1,len(SLR1.loss)+1),SLR1.loss,label='loss',linewidth=5)
plt.plot(np.arange(1,len(SLR1.val_loss)+1),SLR1.val_loss,label='val_loss',linewidth=5)
plt.legend()
plt.show()


# In[92]:


# sklearnで試す
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(train_X1sd,train_y)
y_pred=reg.predict(train_X1sd)


# In[93]:


reg.intercept_


# In[94]:


reg.coef_


# In[95]:


from sklearn.metrics import mean_squared_error
mean_squared_error(train_y, y_pred)


# 【問題1】仮定関数

# In[212]:


# パラメータベクトルは仮でnp.arangeで作成する,そこに本来はパラメータの推定量が入る

def _linear_hypothesis(self, X):
    """
    線形の仮定関数を計算する
    Parameters
    ----------
    X : 次の形のndarray, shape (n_samples, n_features)
      訓練データ
    Returns
    -------
    y :次の形のndarray, shape (n_samples, 1)
       線形の仮定関数による推定結果
    """
    theta=np.arange(1,len(X)+1)
    theta=theta.reshape(len(X),1)
    y=x@theta

    return y


# 【問題2】最急降下法・・・雛形の_gradient_descentで制作

# 【問題3】推定・・・雛形のpredictで制作

# 【問題4】平均二乗誤差・・・雛形の_gradient_descentの中でMSEを計算

# In[220]:


def MSE(y_pred, y):
    """
    平均二乗誤差の計算
    Parameters
    ----------
    y_pred : 次の形のndarray, shape (n_samples,)
      推定した値
    y : 次の形のndarray, shape (n_samples,)
      正解値
    Returns
    ----------
    mse : numpy.float
      平均二乗誤差
    """
    e_sum=0
    for i in range(len(y_pred)):
        e_sum += (y_pred[i]-y[i])**2
    
    mse=e_sum/len(y_pred)
    
    return mse


# 【問題5】目的関数・・・雛形の_gradient_descentの中のlossで計算

# 【問題6】学習と推定・・・上でのsklearn例で示す（値は割と近い）

# 【問題7】学習曲線のプロット・・・上で示した

# 【問題8】（アドバンス課題）バイアス項の除去

# In[254]:


#　まずバイアスがない場合を見てみる

reg3 = LinearRegression(fit_intercept=False)
reg3.fit(train_X,train_y)
reg3.predict(test_X)


# 推定を行うことは可能であるが、１つは定数項がないために評価の１つの尺度である決定係数の解釈に違いが出るため、決定係数は修正をした上で計算し解釈を行わなければならない。

# 【問題9】（アドバンス課題）特徴量の多次元化

# In[261]:


# 特徴量を２乗する

train_XX=train_X*train_X
train_yy=train_y*train_y
test_XX=test_X*test_X
test_yy=test_y*test_y


# In[260]:


reg4 = LinearRegression()
reg3.fit(train_XX,train_yy)
reg3.predict(test_XX)


# In[262]:


test_yy


# 推定は行えるものの、推定結果の残差はより大きくなっているように見える

# In[ ]:




