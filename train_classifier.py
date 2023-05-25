import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


data_dict = pickle.load(open('./onehand.pickle', 'rb'))
data_dict2 = pickle.load(open('./twohand.pickle', 'rb'))
data_one = np.asarray(data_dict['data'])
data_two = np.asarray(data_dict2['data'])
labels1 = np.asarray(data_dict['labels'])
labels2 = np.asarray(data_dict2['labels'])


x_train, x_test, y_train, y_test, = train_test_split(data_one, labels1, test_size=0.2, shuffle=True, stratify=labels1)
x_train2, x_test2, y_train2, y_test2, = train_test_split(data_two, labels2, test_size=0.2, shuffle=True, stratify=labels2)

model1 = RandomForestClassifier()
model2 = RandomForestClassifier()

model1.fit(x_train, y_train)
model2.fit(x_train2, y_train2)

y_predict = model1.predict(x_test)
y_predict2 = model2.predict(x_test2)

score1 = accuracy_score(y_test, y_predict)
score2 = accuracy_score(y_test2, y_predict2)

matrix1 = confusion_matrix(y_test, y_predict)
matrix2 = confusion_matrix(y_test2, y_predict2)

matrix1 = matrix1.astype('float') / matrix1.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix1, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)
class_names = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model (One hand)')
plt.show()

matrix2 = matrix2.astype('float') / matrix2.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix2, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)
class_names = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model (Two hands)')
plt.show()

print('One hand:')
print(classification_report(y_test, y_predict))
print('Two hands:')
print(classification_report(y_test2, y_predict2))

print('{}% of samples were classified correctly for model1 !'.format(score1 * 100))
print('{}% of samples were classified correctly for model2 !'.format(score2 * 100))

f = open('model1.p', 'wb')
pickle.dump({'model1': model1}, f)
f.close()
f = open('model2.p', 'wb')
pickle.dump({'model2': model2}, f)
f.close()
