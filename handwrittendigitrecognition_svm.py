#import the necessary packages
import numpy as np
from sklearn.datasets import load_digits

#load the dataset
dataset = load_digits()
# print(dataset.target)
# print(dataset.data.shape)
# print(dataset.target.shape)
dataiamgeLength=len(dataset.images)
# print(dataiamgeLength)

# visualize the dataset

# n=9 #no of sample out   of samples total 1797
# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(dataset.images[n])
# plt.show()
# dataset.iamges[n]

#Segregate the Y and X
X=dataset.images.reshape((dataiamgeLength,-1))
# print(X)
Y=dataset.target
# print(Y)

#Split the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print(X_train.shape)
# print(X_test.shape)

#Trainng the SVM
from sklearn.svm import SVC
model_svm = SVC(kernel='linear')
model_svm.fit(X_train, Y_train)

#Predict the testing data
Y_pred = model_svm.predict(X_test)
# print(Y_pred)
# print(np.concatenate(( Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)), 1))

#Evaluation - Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy of the model:{0}%".format(accuracy_score(Y_test, Y_pred)*100))