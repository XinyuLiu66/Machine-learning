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



# total have 20 feature, each is a gaussian function g1`````g20,
# so at each data point, data model is [input, output], input:[g1(x)```g20(x)]
def create_data_model_polynomial_feature(num_of_orders):
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


def calculate_polynomial_feature_for_each_data(x, num_of_orders):
    polynomial_features_list = []
    for order in range(1,num_of_orders+1):
        feature = np.power(x, order)
        polynomial_features_list.append(feature)
    polynomial_features_list.append(1)
    return polynomial_features_list

# def calculate_gaussian_feature_for_each_data(x, num_of_features):
#     gaussian_features_list = []
#     miu_sets = np.linspace(0, 2, num_of_features)
#     total_for_normal = 0
#     for miu in miu_sets:
#         gaussian_m = norm(loc=miu, scale=np.sqrt(0.02)).pdf(x)
#         total_for_normal += gaussian_m
#     for miu in miu_sets:
#         gaussian_k = norm(loc=miu,scale=np.sqrt(0.02)).pdf(x)/total_for_normal
#         gaussian_features_list.append(gaussian_k)
#     gaussian_features_list.append(1)
#     return gaussian_features_list


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

def run_model():

    rmse_test_list = []
    rmse_train_list = []

    for i in range(1,21):
        #  train_x_model, test_x_model = create_data_model_gauusian_feature(i)
        W,train_x_model,test_x_model = calculate_w(i)
        print("=======w shape", W.shape)
        #  for test
        y_test_predict = np.matmul(np.transpose(test_x_model), W)
        rmse_test = np.sqrt(mean_squared_error(test_y ,y_test_predict))
        rmse_test_list.append(rmse_test)
        print('===rmse test : ', rmse_test)

        # for train
        y_train_predict = np.matmul(np.transpose(train_x_model), W)
        rmse_train = np.sqrt(mean_squared_error(train_y, y_train_predict))
        rmse_train_list.append(rmse_train)
        print('===rmse train : ', rmse_train)

        if(i == 13):
            plt.scatter(test_x,test_y)
            plt.scatter(test_x,y_test_predict)
            plt.legend(['true', 'predict'])
            plt.show()

    # plt.plot(list(range(1,21)),rmse_test_list,'r')
    # plt.plot(list(range(1,21)),rmse_train_list,'b')
    # plt.legend(['test','train'])
    # plt.show()

run_model()