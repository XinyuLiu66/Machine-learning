import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from numpy import linalg as LA
#==========================Problem 3.3 PCA==========================#

iris_path = "./dataSets/iris.txt"
def load_file(filepath):
    data = loadtxt(filepath, comments="#", unpack=False,delimiter=',')
    return data

data = load_file(iris_path)

#===================normalization of the data to mean = 0, unit variance for each dimension=========

def normalization(data):
    new_data = []
    mean_data = np.mean(data)
    std_var = np.std(data)
    for x in data:
        x = (x - mean_data)/std_var
        new_data.append(x)
    return new_data


sepal_length_data = normalization(data[:,0])
sepal_width_data = normalization(data[:,1])
petal_length_data = normalization(data[:,2])
petal_width_data = normalization(data[:,3])

new_data = np.zeros((len(sepal_length_data), 5))
new_data[:,0] = sepal_length_data
new_data[:,1] = sepal_width_data
new_data[:,2] = petal_length_data
new_data[:,3] = petal_width_data
new_data[:,4] = data[:,4]
# calculate covariance of the matrix for 4 column vector
cov = np.cov([new_data[:,0],new_data[:,1],new_data[:,2],new_data[:,3]])

# print(cov)

# calculate eigenvalue and eigenvectors of covariance matrix

eigenvalue, eigenvector = LA.eig(cov)

#=================b)
fraction_var_list = []
for i in range(1,5):
    sum_covariance = np.sum(eigenvalue[:i])
    fraction_var = sum_covariance/np.sum(eigenvalue)
    if(i == 2):
        print(fraction_var)
    fraction_var_list.append(fraction_var)
plt.plot(list(range(1,5)), fraction_var_list)
plt.show()

#============================c) Low Dimensional Space [6 Points]====================#

projection_matrix = eigenvector[:,:2]
class0_data = []
class1_data = []
class2_data = []

for d in new_data:
    if d[4] == 0.0:
        new_d = d[:4]
        new_d.reshape((1, 4))
        low_dim_data = np.matmul(new_d,projection_matrix)
        class0_data.append(low_dim_data)
    elif d[4] == 1.0:
        new_d = d[:4]
        new_d.reshape((1, 4))
        low_dim_data = np.matmul(new_d,projection_matrix)
        class1_data.append(low_dim_data)
    elif d[4] == 2.0:
        new_d = d[:4]
        new_d.reshape((1, 4))
        low_dim_data = np.matmul(new_d, projection_matrix)
        class2_data.append(low_dim_data)
class0_data = np.array(class0_data)
class1_data = np.array(class1_data)
class2_data = np.array(class2_data)
plt.scatter(class0_data[:,0],class0_data[:,1])
plt.scatter(class1_data[:,0],class1_data[:,1])
plt.scatter(class2_data[:,0],class2_data[:,1])
plt.legend(['class0','class1','class2'])
plt.show()

#============================d) Projection to the Original Space [6 Points]====================#

new_data2 = new_data[:,:4]
def pro_backto_original_space(number_of_components):
    projection_matrix = eigenvector[:, :number_of_components]
    reduced_matrix = np.matmul(new_data2,projection_matrix )
    reconstruct_data = np.matmul(reduced_matrix, np.transpose(projection_matrix))
    return reconstruct_data

#=============normalized root mean square error (NRMSE)========

def NRMSE(original_data, reconstruct_data):
    nrmse_list = []
    for i in range(4):
        feature_set_origin = original_data[:,i]
        feature_set_reconstruct = reconstruct_data[:,i]
        mse = mean_squared_error(feature_set_origin, feature_set_reconstruct)
        nrmse = np.sqrt(mse)/(np.max(feature_set_origin) - np.min(feature_set_origin))
        nrmse_list.append(nrmse)
    return nrmse_list

for i in range(1,5):
    reconstruct_data = pro_backto_original_space(i)
    nrmse = NRMSE(new_data2, reconstruct_data)
    print('the NRMSE of {number} components is {NRMSE}'.format(number=i,NRMSE = nrmse))