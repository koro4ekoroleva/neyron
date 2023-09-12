import numpy as np
import pandas as pd
from pandas import DataFrame

def sigmoid(x):
  # Наша функция активации: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Производная сигмоиды: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true и y_pred - массивы numpy одинаковой длины.
  return ((y_true - y_pred) ** 2).mean()

class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Умножаем входы на веса, прибавляем порог, затем используем функцию активации
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)



# периметр - 15.15
x = np.array(pd.read_excel('сокращённый датасет.xlsx', nrows=648)[['периметр', 'длина ядра', 'ширина ядра']])
y = np.array(pd.read_excel('сокращённый датасет.xlsx', nrows=648)[['класс']])

y_true = y
y_pred = np.array([0]*648)

print(mse_loss(y_true, y_pred))

class OurNeuralNetwork:
  '''
  Нейронная сеть с:
    - 2 входами
    - скрытым слоем с 2 нейронами (h1, h2)
    - выходной слой с 1 нейроном (o1)

  *** DISCLAIMER ***:
  Следующий код простой и обучающий, но НЕ оптимальный.
  Код реальных нейронных сетей совсем на него не похож. НЕ копируйте его!
  Изучайте и запускайте его, чтобы понять, как работает эта нейронная сеть.
  '''
  def __init__(self):
    # Веса
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()
    self.w7 = np.random.normal()
    self.w8 = np.random.normal()
    self.w9 = np.random.normal()
    self.w10 = np.random.normal()
    self.w11 = np.random.normal()
    self.w12 = np.random.normal()

    # Пороги
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()
    self.b4 = np.random.normal()

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
    h2 = sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
    h3 = sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3)
    o1 = sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4)
    return o1

  def train(self, data, all_y_trues):
    '''
    - data - массив numpy (n x 2) numpy, n = к-во наблюдений в наборе.
    - all_y_trues - массив numpy с n элементами.
      Элементы all_y_trues соответствуют наблюдениям в data.
    '''
    learn_rate = 0.1
    epochs = 1000 # сколько раз пройти по всему набору данных

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Прямой проход (эти значения нам понадобятся позже)
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2
        h2 = sigmoid(sum_h2)

        sum_h3 = self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3
        h3 = sigmoid(sum_h3)

        sum_o1 = self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- Считаем частные производные.
        # --- Имена: d_L_d_w1 = "частная производная L по w1"
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Нейрон o1
        d_ypred_d_w10 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w11 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_w12 = h3 * deriv_sigmoid(sum_o1)
        d_ypred_d_b4 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w10 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w11 * deriv_sigmoid(sum_o1)
        d_ypred_d_h3 = self.w12 * deriv_sigmoid(sum_o1)

        # Нейрон h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_w3 = x[2] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Нейрон h2
        d_h2_d_w4 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w5 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_w6 = x[2] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # Нейрон h3
        d_h3_d_w7 = x[0] * deriv_sigmoid(sum_h3)
        d_h3_d_w8 = x[1] * deriv_sigmoid(sum_h3)
        d_h3_d_w9 = x[2] * deriv_sigmoid(sum_h3)
        d_h3_d_b3 = deriv_sigmoid(sum_h3)

        # --- Обновляем веса и пороги
        # Нейрон h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Нейрон h2
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Нейрон h3
        self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w7
        self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w8
        self.w9 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w9
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_b3

        # Нейрон o1
        self.w10 -= learn_rate * d_L_d_ypred * d_ypred_d_w10
        self.w11 -= learn_rate * d_L_d_ypred * d_ypred_d_w11
        self.w12 -= learn_rate * d_L_d_ypred * d_ypred_d_w12
        self.b4 -= learn_rate * d_L_d_ypred * d_ypred_d_b4

      # --- Считаем полные потери в конце каждой эпохи
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))

# Определим набор данных
data = np.array([
  [-0.8, 5.388, 3.377],  # 37
  [-0.47, 5.712, 3.328],   # 53
  [0.94, 6.173, 3.651],   # 98
  [1.48, 6.369, 3.681], # 126
])
all_y_trues = np.array([
  0, # 37
  0, # 53
  1, # 98
  1, # 126
])

# Обучаем нашу нейронную сеть!
network = OurNeuralNetwork()
network.train(data, all_y_trues)

