#pandas - для обработки и анализа данных
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from mlxtend.plotting import plot_confusion_matrix
drive.mount('/content/driver')


#Создание датасета на основе трафика сгенерированного с помощью программы (мой трафик) 1:1:1:1:1 (50000)
base_path = '/content/driver/MyDrive/KNN/my_csv_1_1_1_1_1_50000/'
print(os.listdir(base_path))
case = list()
for file in os.listdir(base_path):
  print(file)
  read_file = pandas.read_csv(base_path + file, header=None)
  case.append(read_file)
  print(read_file.shape)


#Создание датасета на основе трафика сгенерированного с помощью программы (мой трафик) 1:1:1:1 (50000)
base_path = '/content/driver/MyDrive/KNN/my_csv_1_1_1_1_50000/'
print(os.listdir(base_path))
case = list()
for file in os.listdir(base_path):
  print(file)
  read_file = pandas.read_csv(base_path + file, header=None)
  case.append(read_file)
  print(read_file.shape)


#Создание датасета на основе трафика сгенерированного с помощью программы (iot_instrusion) 1:1:1:1 (500)
base_path = '/content/driver/MyDrive/KNN/csv_1_1_1_1_500/'
print(os.listdir(base_path))
case = list()
for file in os.listdir(base_path):
  print(file)
  read_file = pandas.read_csv(base_path + file, header=None)
  case.append(read_file)
  print(read_file.shape)


#Создание датасета на основе трафика сгенерированного с помощью программы (iot_instrusion) 1:1:1:1 (2500)
base_path = '/content/driver/MyDrive/KNN/csv_1_1_1_1_2500/'
print(os.listdir(base_path))
case = list()
for file in os.listdir(base_path):
  print(file)
  read_file = pandas.read_csv(base_path + file, header=None)
  case.append(read_file)
  print(read_file.shape)


#Создание датасета на основе трафика сгенерированного с помощью программы (iot_instrusion) 1:1:1:1 (25000)
base_path = '/content/driver/MyDrive/KNN/csv_1_1_1_1_25000/'
print(os.listdir(base_path))
case = list()
for file in os.listdir(base_path):
  print(file)
  read_file = pandas.read_csv(base_path + file, header=None)
  case.append(read_file)
  print(read_file.shape)

#Создание нового датасета на основе трафика сгенерированного с помощью программы (iot_instrusion) 1:1:1:1 (25000)
base_path = '/content/driver/MyDrive/KNN/new_1_1_1_1_25000/'
print(os.listdir(base_path))
case = list()
for file in os.listdir(base_path):
  print(file)
  read_file = pandas.read_csv(base_path + file, header=None)
  case.append(read_file)
  print(read_file.shape)

#Создание датасета на основе трафика сгенерированного с помощью программы (iot_instrusion) 4:4:1:1 (25000)
base_path = '/content/driver/MyDrive/KNN/csv_4_4_1_1_25000/'
print(os.listdir(base_path))
case = list()
for file in os.listdir(base_path):
  print(file)
  read_file = pandas.read_csv(base_path + file, header=None)
  case.append(read_file)
  print(read_file.shape)

#Создание датасета на основе трафика сгенерированного с помощью программы (iot_instrusion) 1:8:1:1 (25000)
base_path = '/content/driver/MyDrive/KNN/csv_1_8_1_1_25000/'
print(os.listdir(base_path))
case = list()
for file in os.listdir(base_path):
  print(file)
  read_file = pandas.read_csv(base_path + file, header=None)
  case.append(read_file)
  print(read_file.shape)


#Создание датасета на основе трафика сгенерированного с помощью программы (iot_instrusion) 7:1:7:7 (25000)
base_path = '/content/driver/MyDrive/KNN/csv_7_1_7_7_25000/'
print(os.listdir(base_path))
case = list()
for file in os.listdir(base_path):
  print(file)
  read_file = pandas.read_csv(base_path + file, header=None)
  case.append(read_file)
  print(read_file.shape)


#Создание датасета на основе трафика сгенерированного с помощью программы (iot_instrusion) 1:1:1:1 (40000)
base_path = '/content/driver/MyDrive/KNN/csv_1_1_1_1_40000/'
print(os.listdir(base_path))
case = list()
for file in os.listdir(base_path):
  print(file)
  read_file = pandas.read_csv(base_path + file, header=None)
  case.append(read_file)
  print(read_file.shape)

#Создание датасета на основе трафика сгенерированного с помощью программы (iot_instrusion) 1:1:1:1 (100000)
base_path = '/content/driver/MyDrive/KNN/csv_1_1_1_1_100000/'
print(os.listdir(base_path))
case = list()
for file in os.listdir(base_path):
  print(file)
  read_file = pandas.read_csv(base_path + file, header=None)
  case.append(read_file)
  print(read_file.shape)


#Создание датасета на основе трафика сгенерированного с помощью программы (iot_instrusion) 1:1:1:1 (180000)
base_path = '/content/driver/MyDrive/KNN/csv_1_1_1_1_180000/'
print(os.listdir(base_path))
case = list()
for file in os.listdir(base_path):
  print(file)
  read_file = pandas.read_csv(base_path + file, header=None)
  case.append(read_file)
  print(read_file.shape)


#Создание датасета на основе трафика сгенерированного с помощью программы (iot_instrusion) RANDOM
base_path = '/content/driver/MyDrive/KNN/csv_random/'
print(os.listdir(base_path))
case = list()
for file in os.listdir(base_path):
  print(file)
  read_file = pandas.read_csv(base_path + file, header=None)
  case.append(read_file)
  print(read_file.shape)


# Обзор dataset
data = pandas.concat(case)
print('\n',data)
y = data[13]
print(y)
y = y.to_numpy().astype('int')
print(y)
data.drop([13], axis='columns', inplace=True)
print(data)
X = data.to_numpy().astype('float')
print(X)


# Разделение датасета на обучающую и тестовую выборки
# 9:1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=12345)
# 7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)
# 5:5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12345)
# 3:7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=12345)
# 1:9
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=12345)


# Длина выборок
print("Общая длина: ",len(y_train) + len(y_test))
print("Обучающая : ",len(X_train))
print("Тестовая : ",len(X_test))


# Поиск K от 3 до 21
m = 21
# m = 101
r = []
for k in np.arange(3,m,2):
  print(k)
  new_model = KNeighborsClassifier(n_neighbors=k)
  new_model.fit(X_train, y_train)
  new_predictions = new_model.predict(X_test)
  r.append(precision_score(y_test, new_predictions, average='weighted'))
nghb = 2 * np.argmax(r) + 3
plt.plot(np.arange(3,m,2), r)
print('Лучший показатель precision:', max(r))
print('При k = ',nghb)


# Выбор количества соседей
n_neighbors = 3
# n_neighbors = 5
# n_neighbors = 7
# n_neighbors = 9
# n_neighbors = 17


# Обучение
model = KNeighborsClassifier(n_neighbors)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probs = model.predict_proba(X_test)


# Расчёт метрики
print("Precision:", precision_score(y_test, predictions, average='weighted'))
print("Accuracy:", accuracy_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions, average='macro'))
print("F1 score:", f1_score(y_test,predictions, average='weighted'))
print("ROC:",roc_auc_score(y_test, probs, multi_class='ovr'))


print("Precision:", precision_score(y_test, predictions, average='weighted'))
print("Accuracy:", accuracy_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions, average='macro'))
print("F1 score:", f1_score(y_test,predictions, average='weighted'))
print("ROC:",roc_auc_score(y_test, probs, multi_class='ovr'))
