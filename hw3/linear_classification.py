import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.metrics import mean_squared_error


ldaData_path = "./dataSets/ldaData.txt"
def load_file(filepath):
    data = loadtxt(filepath, comments="#", unpack=False)
    return data

data = load_file(ldaData_path)
class1_data = data[:50]
class2_data = data[50:100]
class3_data = data[100:150]


# plt.scatter(class1_data[:,0],class1_data[:,1])
# plt.scatter(class2_data[:,0],class2_data[:,1])
# plt.scatter(class3_data[:,0],class3_data[:,1])
# plt.legend(['Class1', 'Class2', 'Class3'])
# # plt.show()


#=================compute mean of x for each class===================#

# shape of class_data_x is (#samples, #features)
def comput_mean_x(class_x_model):
    return np.mean(class_x_model, axis=0)

#=================compute varience of x for each class===================#

def comput_covariance_in_a_class(class_x_model):
    mean_x_model = comput_mean_x(class_x_model)
    covar_class_x_model = np.zeros((len(mean_x_model), len(mean_x_model)),dtype=np.float32)

    for x in class_x_model:
        x_minus_mean = x - mean_x_model
        x_minus_mean = x_minus_mean.reshape((len(x_minus_mean),1))
        covar_each_x = np.matmul(x_minus_mean, np.transpose(x_minus_mean))
        covar_class_x_model += covar_each_x
    return covar_class_x_model, mean_x_model

#==========================compute W =======================#
def compute_projection_vector_W(class_1, class_2):
    cov_class_1,m1 = comput_covariance_in_a_class(class_1)
    cov_class_2,m2 = comput_covariance_in_a_class(class_2)
    SW = cov_class_1 + cov_class_2
    W = np.matmul(np.linalg.inv(SW), (m1 - m2))
    print(W.shape)
    return W

#==========================projection=======================#
def projection(x1,x2, k):
    b = 0
    X = np.linspace(0,2,60)
    Y = k * X + b
    # 计算投影点M(x1, y1)
    pro_x = (k * (x2 - b) + x1) / (k * k + 1);
    pro_y = k * pro_x + b;
    pro_point = [pro_x, pro_y]
    return pro_point


def run_model(class_1, class_2):
    y_list_class1 = []
    y_list_class2 = []
    pro_class1_points_set = []
    pro_class2_points_set = []
    pro_class3_points_set = []
    W = compute_projection_vector_W(class_1, class_2)
    k = W[1]/W[0]
    mo_W = np.sqrt(W[0]**2 + W[1]**2)
    for x in class_1:
        x1 = x[0]
        x2 = x[1]
        pro_class1_data = projection(x1,x2, k)
        pro_class1_points_set.append(pro_class1_data)
        y = np.dot(W,x)
        y = y/ mo_W
        y_list_class1.append(y)
    y_list_class1 = np.array(y_list_class1)
    pro_class1_points_set = np.array(pro_class1_points_set)
    for x in class_2:
        x_1 = x[0]
        x_2 = x[1]
        pro_class2_data = projection(x_1,x_2, k)
        pro_class2_points_set.append(pro_class2_data)
        y = np.dot(W,x)
        y = y/ mo_W
        y_list_class2.append(y)
    y_list_class2 = np.array(y_list_class2)
    pro_class2_points_set = np.array(pro_class2_points_set)
    # plt.plot(pro_class1_points_set[:,0], pro_class1_points_set[:,1],"r*")
    # plt.plot(pro_class2_points_set[:,0], pro_class2_points_set[:,1],'b+')
    # plt.plot(y_list_class1,np.zeros(len(y_list_class1)),'r*')
    # plt.plot(y_list_class2, np.zeros(len(y_list_class1)), 'b+')
    return k, W

#================create classification according to got projection================#

# re_classify class2 and class 3
def re_classify(datas, threthhold, W):
    class2_new_datasets = []
    class3_new_datasets = []
    mo_W = np.sqrt(W[0] ** 2 + W[1] ** 2)
    y_list = []
    count_misclassification = 0
    for i, data in enumerate(datas):
        y = np.dot(W, data)/mo_W
        y_list.append(y)
        if(y > threthhold):
            class2_new_datasets.append(data)
            if (i > 50):
                count_misclassification += 1
        else:
            class3_new_datasets.append(data)
            if(i < 50):
                count_misclassification += 1
    class2_new_datasets = np.array(class2_new_datasets)
    class3_new_datasets = np.array(class3_new_datasets)
    return class2_new_datasets , class3_new_datasets, count_misclassification



# k1 = run_model(class1_data, class2_data)
k2,W2 = run_model(class2_data, class3_data)
class2_new_datasets,class3_new_datasets,count_misclassification = re_classify(data[50:], -6.8, W2)


print("the number of  misclassified points ",count_misclassification )
plt.scatter(class1_data[:,0],class1_data[:,1])
plt.scatter(class2_new_datasets[:,0],class2_new_datasets[:,1])
plt.scatter(class3_new_datasets[:,0],class3_new_datasets[:,1])
plt.legend(['Class1', 'Class2', 'Class3'])
# # plt.show()

#plt.plot(x,k1*x)
#plt.plot(x,k2*x)

plt.show()

