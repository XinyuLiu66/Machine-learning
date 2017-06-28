import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.metrics import mean_squared_error


linRegData_path = "./dataSets/linRegData.txt"
def load_file(filepath):
    data = loadtxt(filepath, comments="#", unpack=False)
    return data

linRegData = load_file(linRegData_path)

#============================create_train_test_data_model========================
def create_train_test_data_model(k):
    train_data = linRegData[:k]
    test_data = linRegData[k:]
    test_data = test_data[np.lexsort(np.fliplr(test_data).T)]
    # x = linRegData[:,0]
    # y = linRegData[:,1]
    train_x = train_data[:,0].reshape((k,1))
    train_y = train_data[:,1].reshape((k,1))
    test_x =  test_data[:,0].reshape((150-k,1))
    test_y =  test_data[:,1].reshape((150-k,1))
    return  train_x, train_y, test_x, test_y

# total have 20 feature, each is a gaussian function g1`````g20,
# so at each data point, data model is [input, output], input:[g1(x)```g20(x)]
train_x, train_y, test_x, test_y = create_train_test_data_model(149)

#============================create_train_test_data_model========================

def create_data_model_polynomial_feature(num_of_orders = 12):
    trainData_feature_sets = []
    testData_feature_sets = []

    # for train data sets
    for x_train_value in train_x:
        feature_train_sets = calculate_polynomial_feature_for_each_data(float(x_train_value),num_of_orders)
        trainData_feature_sets.append(feature_train_sets)
    train_x_model = np.transpose(np.array(trainData_feature_sets))
    print(train_x_model.shape)

    # for test data sets
    for x_value in test_x:
        feature_sets = calculate_polynomial_feature_for_each_data(float(x_value),num_of_orders)
        testData_feature_sets.append(feature_sets)
    test_x_model = np.transpose(np.array(testData_feature_sets))
    print(test_x_model.shape)
    return train_x_model, test_x_model

#==============================calculate polynomial features===============================

def calculate_polynomial_feature_for_each_data(x, num_of_orders):
    polynomial_features_list = []
    for order in range(1,num_of_orders+1):
        feature = np.power(x, order)
        polynomial_features_list.append(feature)
    polynomial_features_list.append(1)
    return polynomial_features_list

#==============================calculate W===================================

def calculate_w(num_of_features):
    train_x_model, test_x_model = create_data_model_polynomial_feature(num_of_features)
   # part1 = np.linalg.inv(np.matmul(train_x_model, np.transpose(train_x_model)) )
    part1 = np.matmul(train_x_model, np.transpose(train_x_model))
    shape_part1 = part1.shape
    regular_unit_matrix = np.identity(shape_part1[0])
    part1 = part1 + 0.000006*regular_unit_matrix
    part1 = np.linalg.inv(part1)
    part2 = np.matmul(part1, train_x_model)
    W = np.matmul(part2, train_y)

    #W = np.matmul(  np.matmul(   np.linalg.inv(  np.matmul(train_x_model, np.transpose(train_x_model) )),train_x_model), train_y)
    return W , train_x_model, test_x_model

#==============================calculate covariance matrix of predicted distribution ===========
# p(yt|xt, D) ~ N(y| u, 1/lamda),   u = (beta * 1/M * X_T * y)_T * X, M = beta *X_T *X + alpha * I
# 1/lamda = 1/beta + X_T * 1/M * X

def calculate_predicted_value_deviation(nuber_of_features):
    alpha = 20 * 0.000006
    beta = 20
    W, train_x_model, test_x_model = calculate_w(nuber_of_features)
    part1 =  np.matmul(train_x_model , np.transpose(train_x_model))
    shape_part1 = part1.shape
    regular_unit_matrix = np.identity(shape_part1[0])

    # precision_matrix is the 1/(covariance matrix)
    precision_matrix = beta * part1 + alpha * regular_unit_matrix
    deviation_for_test_model_list = []
    for test_input in np.transpose(test_x_model):
        deviation_for_each_testdata_sample = (1 / beta) + np.matmul(np.matmul(test_input, np.linalg.inv(precision_matrix)),np.transpose(test_input))
        deviation_for_test_model_list.append(deviation_for_each_testdata_sample)
    return deviation_for_test_model_list

#===============================run model===================================

def run_model():

    #  train_x_model, test_x_model = create_data_model_gauusian_feature(i)
    W,train_x_model,test_x_model = calculate_w(12)
    print("=======w shape", W.shape)
    #  for test
    y_mean_test_predict = np.matmul(np.transpose(test_x_model), W)

    deviation_for_test_model_list = calculate_predicted_value_deviation(12)
    _len = len(deviation_for_test_model_list)
    deviation_for_test_model = np.array(deviation_for_test_model_list).reshape((_len, 1))
    # upper bound of predicetd y
    y_upper_test_predict = y_mean_test_predict + deviation_for_test_model

    # lower bound of predicetd y
    y_lower_test_predict = y_mean_test_predict - deviation_for_test_model




    plt.scatter(test_x,test_y)
    plt.scatter(test_x,y_mean_test_predict)
    plt.scatter(test_x, y_lower_test_predict)
    plt.legend(['mean', 'uppper bound',  'lower bound'])
    plt.ylim([-2.5,2.5])
    #plt.yticks(np.arange(-10,10.5))
    plt.show()

    # plt.plot(list(range(1,17)),rmse_test_list,'r')
    # plt.plot(list(range(1,17)),rmse_train_list,'b')
    # plt.legend(['test','train'])
    # plt.show()

run_model()