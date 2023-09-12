import numpy as np
import pandas as pd
from pandas import DataFrame

x = np.array(pd.read_excel('D:\ЛИЗА\БГИТУ\нейронка\сокращённый датасет.xlsx', nrows=648)[['периметр', 'длина ядра', 'ширина ядра']])
y = np.array(pd.read_excel('D:\ЛИЗА\БГИТУ\нейронка\сокращённый датасет.xlsx', nrows=648)[['класс']])

def mse_loss(y_true, y_pred):
  # y_true и y_pred - массивы numpy одинаковой длины.
  return ((y_true - y_pred) ** 2).mean()

y_true = y
y_pred = np.array([0]*648)

print(y)

#print(mse_loss(y_true, y_pred))