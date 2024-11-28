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
import matplotlib.pyplot as plt
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
model_svm = SVC( )
model_svm.fit(X_train, Y_train)
print(model_svm.kernel)
# #Predict the testing data
# Y_pred = model_svm.predict(X_test)
# # print(Y_pred)
# # print(np.concatenate(( Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)), 1))

# #Evaluation - Accuracy
# from sklearn.metrics import accuracy_score
# print("Accuracy of the model:{0}%".format(accuracy_score(Y_test, Y_pred)*100))


# #Prediction
# n=15
# result = model_svm.predict(dataset.images[n].reshape((1,-1)))
# plt.imshow(dataset.images[n], cmap=plt.cm.gray_r, interpolation='nearest')     
# print(result)
# print("\n")
# plt.axis('off')
# plt.title('%i' %result)
# plt.show()






#playing kernal with different values
# model1=SVC(kernel='linear')
# model2=SVC(kernel='rbf')
# model3=SVC(kernel='poly')
# model4=SVC(kernel='sigmoid')
# model5=SVC()
# model6=SVC(gamma=0.001)
# model7=SVC(gamma=0.001 ,C=0.1)

# model1.fit(X_train, Y_train)
# model2.fit(X_train, Y_train)
# model3.fit(X_train, Y_train)
# model4.fit(X_train, Y_train)
# model5.fit(X_train, Y_train)
# model6.fit(X_train, Y_train)
# model7.fit(X_train, Y_train)

# Y_pred1 = model1.predict(X_test)
# Y_pred2 = model2.predict(X_test)
# Y_pred3 = model3.predict(X_test)
# Y_pred4 = model4.predict(X_test)
# Y_pred5 = model5.predict(X_test)
# Y_pred6 = model6.predict(X_test)
# Y_pred7 = model7.predict(X_test)

# print("Accuracy of the model1:{0}%".format(accuracy_score(Y_test, Y_pred1)*100))
# print("Accuracy of the model2:{0}%".format(accuracy_score(Y_test, Y_pred2)*100))
# print("Accuracy of the model3:{0}%".format(accuracy_score(Y_test, Y_pred3)*100))
# print("Accuracy of the model4:{0}%".format(accuracy_score(Y_test, Y_pred4)*100))
# print("Accuracy of the model5:{0}%".format(accuracy_score(Y_test, Y_pred5)*100))
# print("Accuracy of the model6:{0}%".format(accuracy_score(Y_test, Y_pred6)*100))
# print("Accuracy of the model7:{0}%".format(accuracy_score(Y_test, Y_pred7)*100)) 

# Accuracy of the model:97.77777777777777%
# Accuracy of the model1:97.77777777777777%
# Accuracy of the model2:99.16666666666667%
# Accuracy of the model3:98.88888888888889%
# Accuracy of the model4:91.38888888888889%
# Accuracy of the model5:99.16666666666667%
# Accuracy of the model6:99.16666666666667%
# Accuracy of the model7:97.22222222222221%