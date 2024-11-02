#import the necessary packages
import numpy as np
from sklearn.datasets import load_digits

#load the dataset
dataset = load_digits()
# print(dataset.target)
# print(dataset.data.shape)
# print(dataset.target.shape)

# visualize the dataset

n=9 #no of sample out   of samples total 1797
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(dataset.images[n])
plt.show()
dataset.iamges[n]