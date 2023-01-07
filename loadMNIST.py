#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np
import struct
from array import array
from os.path import join
import os
import random
import matplotlib.pyplot as plt
import math


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)  
		
		
#
# Verify Reading Dataset via MnistDataloader class
#
#
# Set file paths based on added MNIST Datasets
#
cwd = os.getcwd()
input_path = cwd + '\MNIST'
training_images_filepath = '/Users/talnesher/Downloads/intoToDataSciece/assignment2IntroToDataScience/train-images.idx3-ubyte'
training_labels_filepath = '/Users/talnesher/Downloads/intoToDataSciece/assignment2IntroToDataScience/train-labels.idx1-ubyte'
test_images_filepath = '/Users/talnesher/Downloads/intoToDataSciece/assignment2IntroToDataScience/t10k-images.idx3-ubyte'
test_labels_filepath = '/Users/talnesher/Downloads/intoToDataSciece/assignment2IntroToDataScience/t10k-labels.idx1-ubyte'

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#
# Show some random training and test images 
#
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

#show_images(images_2_show, titles_2_show)

#__________________a_____________________

#reduce all images to between 0.5 to -0.5
x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = (x_train / 255.0) - 0.5
x_test = (x_test / 255.0) - 0.5
m = x_train.shape[0]

#___________________b__________________

#flattening the images
X = (x_train.reshape(x_train.shape[0], -1)).T

# Compute the covariance matrix    
Theta = np.dot(X, X.T) / m

# Compute the eigendecomposition of the covariance matrix
U, S, Ut = np.linalg.svd(Theta)
S_squared = np.sqrt(S)

# Plot the singular values
# plt.plot(S_squared)
# plt.show()

# Select the first p columns of U as Up
p = 40
Up = U[:,:p]

# Compute the reduced-sized vectors for each image. now each coulm in w represent a picture with 40 elements instead of 784
w = np.dot(Up.T, X)

# Reconstruct an image from the reduced-sized vector
i = 1 # choose an image index
x_reconstructed = np.dot(Up, w[:, i])

# Reshape the reconstructed image back to a 2D array
x_reconstructed = x_reconstructed.reshape(28, 28)

#Plot the original and reconstructed images
# plt.subplot(1, 2, 1)
# plt.imshow(x_train[i], cmap = 'gray')
# plt.title('Original image')
# plt.subplot(1, 2, 2)
# plt.imshow(x_reconstructed, cmap='gray')
# plt.title('Reconstructed image')
# plt.show()

#_________________c_______________

def kmeans(X, k, centers_kmeans ,max_iter = 5):
    for i in range (max_iter):
        assigned_centers = assign_to_closest_center(X, centers_kmeans)
        centers_kmeans = recompute_centers(X, assigned_centers, k, centers_kmeans)
        plt.scatter(X[:, 0], X[:, 1], c = assigned_centers, cmap = 'viridis')
        plt.scatter(centers_kmeans[:, 0], centers_kmeans[:, 1], c = 'red', marker = 'x')
        plt.show()
    return assigned_centers,centers_kmeans


def assign_to_closest_center(B, centers_assign):
  assigned_centers = []
  # Loop through each photo
  for d in range(B.shape[0]):
    # Calculate the distances between the photo and each center
    photo=B[d]
    distances = []
    for i in range(centers_assign.shape[0]):
      temp_center = centers_assign[i]
      distance = 0
      for j in range(B.shape[1]):
        distance += (photo[j] - temp_center[j]) ** 2
      distance = math.sqrt(distance)
      distances.append(distance)
    
    # Find the index of the center with the shortest distance
    min_index = distances.index(min(distances))
    
    # Assign the center with the shortest distance to the photo
    assigned_centers.append(min_index)
  return assigned_centers

def recompute_centers(B, assigned_centers, k, centers):
    numOfelements = B.shape[1]
    assigned_centers_array = np.array(assigned_centers)
    numOfImagesPerCenter = np.zeros(k)
    sumOfVectors = np.zeros((k, numOfelements))
    for i in range(len(assigned_centers_array)):
        center = assigned_centers_array[i]
        numOfImagesPerCenter[center] += 1
        for p in range(numOfelements):
            sumOfVectors[center,p] += B[i,p]
    for j in range (k):
        if numOfImagesPerCenter[j]!=0:
            sumOfVectors[j] = sumOfVectors[j] / numOfImagesPerCenter[j]
        else:
            sumOfVectors[j]=centers[j]
    return sumOfVectors


def find_max_index(arr):
    max_index = 0
    for i in range(1, len(arr)):
        if arr[i] > arr[max_index]:
            max_index = i
    return max_index


#run kmean with p=40 and randomize centers
k = 10
# Initialize the centroids randomly
centroids = np.random.uniform(low=-0.5, high=0.5, size=(k, w.T.shape[1]))
assigned_centers,centers1 = kmeans(w.T, k, centroids)