import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
#==========================Problem 3.1 Linear Regression==========================#


#================= a) Polynomial Features [10 Points]=============#


linRegData_path = "./dataSets/linRegData.txt"
def load_file(filepath):
    data = loadtxt(filepath, comments="#", unpack=False)
    return data

linRegData = load_file(linRegData_path)
#linRegData = linRegData[np.lexsort(np.fliplr(linRegData).T)]
train_data = linRegData[:20]
test_data = linRegData[20:]
test_data = test_data[np.lexsort(np.fliplr(test_data).T)]
# x = linRegData[:,0]
# y = linRegData[:,1]
train_x = train_data[:,0].reshape((20,1))
train_y = train_data[:,1].reshape((20,1))
test_x =  test_data[:,0].reshape((130,1))
test_y =  test_data[:,1].reshape((130,1))

 # plot
# fig = plt.figure()
# plt.scatter(test_x, test_y)
# plt.show()
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import  PolynomialFeatures


# def create_polynomial_feature_model():
#     k_mse_train = []
#     k_mse_test = []
#     for k in range(1,26):
#         mse_train, mse_test = k_Polynomial_features(k)
#         k_mse_train.append(mse_train)
#         k_mse_test.append(mse_test)
#
#
#     #plot score
#     print("====train mse======")
#     for i, mse_train in enumerate(k_mse_train):
#         print(i+1,'  ',mse_train)
#     print("====test mse======")
#     for i, mse_test in enumerate(k_mse_test):
#         print(i + 1, '  ', mse_test)
#     plt.plot(list(range(1,26)), k_mse_train,'b',list(range(1,26)), k_mse_test,'r')
#     plt.legend(['train data','test data'])
#     plt.xlabel("number of features")
#     plt.ylabel("rmse")
#     plt.show()
#    # plt.plot(test_x,regression_k.predict(test_x))
# #===================================================#
# def k_Polynomial_features(k):
#
#     if k == 1:
#         regressor = LinearRegression()
#         regressor.fit(train_x, train_y)
#
#         # for test data
#         predicted_test = regressor.predict(test_x)
#         mse_test = np.sqrt(mean_squared_error(predicted_test, test_y))
#
#         # for train data
#         predicted_train = regressor.predict(train_x)
#         mse_train = np.sqrt(mean_squared_error(predicted_train, train_y))
#
#     else:
#         k_featurizer = PolynomialFeatures(degree=k)
#         X_train_k = k_featurizer.fit_transform(train_x)
#         X_test_k = k_featurizer.transform(test_x)
#         regression_k = LinearRegression()
#         regression_k.fit(X_train_k, train_y)
#
#         # for test data
#         predicted_test = regression_k.predict(X_test_k)
#         mse_test = np.sqrt(mean_squared_error(predicted_test, test_y))
#
#         # for train data
#         predicted_train = regression_k.predict(X_train_k)
#         mse_train = np.sqrt(mean_squared_error(predicted_train, train_y))
#
#     if(k == 26):
#         plt.plot(test_x, test_y,'b^')
#         plt.plot(test_x, regression_k.predict(X_test_k),'r--')
#         plt.legend(['true data', 'predict data'])
#         plt.show()
#
#     return mse_train, mse_test
#
#
# create_polynomial_feature_model()


#===============b) Gaussian Features [4 Points]=====================

def gaussian_features(k,sigma): # k number of features
    miu_sets = np.linspace(0, 2, k)
    x_axis = np.arange(0, 2, 0.01)

    gaussain_function = 0
    for miu in miu_sets:
        gaussain_function_k = norm(loc=miu,scale=sigma).pdf(x_axis)
        gaussain_function += gaussain_function_k
    for miu in miu_sets:
        gaussain_function_m = norm(loc=miu,scale=sigma)
        y_m = gaussain_function_m.pdf(x_axis)/gaussain_function
        plt.plot(x_axis,y_m)
    plt.show()


gaussian_features(20,np.sqrt(0.02))


#===============c) Gaussian Features, Continued [6 Points]=====================

# total have 20 feature, each is a gaussian function g1`````g20,
# so at each data point, data model is [input, output], input:[g1(x)```g20(x)]
def create_data_model_gauusian_feature(num_of_features):
    trainData_feature_sets = []
    testData_feature_sets = []

    # for train data sets
    for x_train_value in train_x:
        feature_train_sets = calculate_gaussian_feature_for_each_data(float(x_train_value),num_of_features)
        trainData_feature_sets.append(feature_train_sets)
    train_x_model = np.transpose(np.array(trainData_feature_sets))
    print(train_x_model.shape)

    # for test data sets
    for x_value in test_x:
        feature_sets = calculate_gaussian_feature_for_each_data(float(x_value),num_of_features)
        testData_feature_sets.append(feature_sets)
    test_x_model = np.transpose(np.array(testData_feature_sets))
    print(test_x_model.shape)
    return train_x_model, test_x_model



def calculate_gaussian_feature_for_each_data(x, num_of_features):
    gaussian_features_list = []
    miu_sets = np.linspace(0, 2, num_of_features)
    total_for_normal = 0
    for miu in miu_sets:
        gaussian_m = norm(loc=miu, scale=np.sqrt(0.02)).pdf(x)
        total_for_normal += gaussian_m
    for miu in miu_sets:
        gaussian_k = norm(loc=miu,scale=np.sqrt(0.02)).pdf(x)/total_for_normal
        gaussian_features_list.append(gaussian_k)
    gaussian_features_list.append(1)
    return gaussian_features_list


# Test
# def gaussian(x,mu,sig):
#     return np.exp(-np.power((x - mu),2.)/(2* np.power(sig,2.0)))/(np.sqrt(2*np.pi) * sig)
#
# print(calculate_gaussian_feature_for_each_data(0, 3))
# print(gaussian(0,0,np.sqrt(0.02))/(gaussian(0,0,np.sqrt(0.02)) + gaussian(0,1,np.sqrt(0.02))+ gaussian(0,2,np.sqrt(0.02))))
# print(gaussian(0,1,np.sqrt(0.02)))
# print(gaussian(0,2,np.sqrt(0.02)))



#

def calculate_w(i):
    train_x_model, test_x_model = create_data_model_gauusian_feature(i)
    part1 = np.linalg.inv(np.matmul(train_x_model, np.transpose(train_x_model)))
    part2 = np.matmul(part1, train_x_model)
    W = np.matmul(part2, train_y)

    #W = np.matmul(  np.matmul(   np.linalg.inv(  np.matmul(train_x_model, np.transpose(train_x_model) )),train_x_model), train_y)
    return W , train_x_model, test_x_model

def run_model():

    rmse_test_list = []
    rmse_train_list = []

   # for i in range(2,20):
        #  train_x_model, test_x_model = create_data_model_gauusian_feature(i)
    W,train_x_model,test_x_model = calculate_w(30)
    print("=======w shape", W.shape)
    #  for test
    y_test_predict = np.matmul(np.transpose(test_x_model), W)
    rmse_test = np.sqrt(mean_squared_error(test_y ,y_test_predict))
    print(rmse_test)

   #  # for train
   #  y_train_predict = np.matmul(np.transpose(W),train_x_model)
   #  rmse_train = np.sqrt(mean_squared_error( np.transpose(y_train_predict), train_y))
   # # rmse_test_list.append(rmse_test)
   #  print(rmse_train)

    plt.scatter(test_x,test_y)
    plt.scatter(test_x[100:],y_test_predict[100:])
    plt.legend(['true', 'predict'])
    plt.show()

    # plt.plot(list(range(15, 41)), rmse_test_list, 'r')
    # plt.show()

run_model()


# def train_model():
#     mse_test_list = []
#     mse_train_list = []
#     for i in range(15,41):
#         train_x_model, test_x_model = create_data_model_gauusian_feature(i)
#
#         regressor = LinearRegression()
#         regressor.fit(train_x_model, train_y)
#
#         # for test data
#         predicted_test = regressor.predict(test_x_model)
#         mse_test = np.sqrt(mean_squared_error(predicted_test, test_y))
#         mse_test_list.append(mse_test)
#
#         # for train data
#         predicted_train = regressor.predict(train_x_model)
#         mse_train = np.sqrt(mean_squared_error(predicted_train, train_y))
#         mse_train_list.append(mse_train)
#
#     # plot rmse for traing and test data sets
#
#     plt.plot(list(range(15,41)),mse_train_list,'b',list(range(15,41)),mse_test_list,'r')
#     plt.legend(['trainning data','test data'])
#     plt.show()
#
#
#     print("=========mse_test   ", mse_test )
#
# train_model()
