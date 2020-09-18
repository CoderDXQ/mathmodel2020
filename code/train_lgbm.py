#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
import pylab as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer

X = pd.read_csv("processed.csv")
y = X.pop("RON_loss")
X_ = QuantileTransformer(n_quantiles=1000).fit_transform(X)
selector = SelectFromModel(
    estimator=GradientBoostingRegressor(random_state=0))
# 筛一次可能不够好，要筛两次
X_ = selector.fit_transform(X_, y) # 76 columns left
X_ = selector.fit_transform(X_, y) # 22 columns left
print(f"两次特征筛选后的X_.shape = {X_.shape}")
pipeline = Pipeline([
    # ("transform", QuantileTransformer(n_quantiles=1000)),
    ("regressor", LGBMRegressor(random_state=0, n_estimators=100, learning_rate=0.1)),
])
cv = KFold(n_splits=5, shuffle=True, random_state=0)
valid_score = cross_val_score(pipeline, X_, y, cv=cv)
print(f"5折交叉验证后，在验证集上的平均r2 = {valid_score.mean()}\n"
      f"每折的r2 = {valid_score.tolist()}")
pipeline.fit(X_, y)
y_pred = pipeline.predict(X_)
train_score=r2_score(y, y_pred)
pearson_correlation = pearsonr(y,y_pred)[0]
print(f"在训练集上，r2 = {train_score}, pearson 相关系数 = {pearson_correlation}")
plt.xlabel("y true")
plt.ylabel("y pred")
plt.scatter(y, y_pred)
plt.show()
