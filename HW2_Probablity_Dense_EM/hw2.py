import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
# path1 = "/Users/apple/Documents/tu_darmstadt/MSC_2semester/statistic_mahine_learning/homework/hw2/dataSets/densEst1.txt"
# path2 = "/Users/apple/Documents/tu_darmstadt/MSC_2semester/statistic_mahine_learning/homework/hw2/dataSets/densEst2.txt"
# def counut_Nr_Rows(path):
#     listdata = []
#     count = 0
#     with open(path) as file:
#         for line in file:
#             listdata.append(line)
#             count += 1
#     return listdata,count
#
# c1_data, Nr_dataset1 = counut_Nr_Rows(path1)
# c2_data, Nr_dataset2 = counut_Nr_Rows(path2)
#
# #======================== Problem 2.2(c) Biased ML Estimate ===================#
#
# print("Nr_C1 = ", Nr_dataset1)
#
# print("Nr_C2 = ", Nr_dataset2)
#
# p_c1 = Nr_dataset1/(Nr_dataset1+Nr_dataset2)
# p_c2 = Nr_dataset2/(Nr_dataset1+Nr_dataset2)
# print("probability of_C1 = ", p_c1)
# print("probability of_C2 = ", p_c2)
#
# #print(c1_data[0])
# #======================== Problem 2.2(c) Biased ML Estimate ===================#
#
# def data_pre_procassing(pre_data):
#     vec_data = []
#     for row in pre_data:
#         row = row.lstrip()
#         str_two_data = row.split("  ")
#         data1 = float(str_two_data[0])
#         data2 = float(str_two_data[1].lstrip())
#         vec_data.append([data1,data2])
#     np_data = np.array(vec_data)
#     return np_data
#
# c1 = data_pre_procassing(c1_data)
# c2 = data_pre_procassing(c2_data)
#
# #====calculate unbias estimator for  expection===
# u_c1 = c1.sum(axis=0)/len(c1)
# u_c2 = c2.sum(axis=0)/len(c2)
# print("u(c1)= ", u_c1)
# print("u(c2)= ", u_c2)
#
# #===calculate unbias estimator for  expection===
#
#
# def unbias_estimator_convarience(c):
#     c_1 = np.array(c[:, 0])
#     c_2 = np.array(c[:, 1])
#     # method1: using external function
#     c_convarience = np.cov(np.transpose(c))
#
#     # method2: calculate by myself
#     cov_c1_11 = np.sum((c_1 - u_c1[0]) * (c_1 - u_c1[0]) / ((len(c)) - 1))
#     cov_c1_22 = np.sum(np.power(c_2 - u_c1[1], 2)) / (len(c1) - 1)
#     cov_c1_12 = np.sum((c_1 - u_c1[0]) * (c_2 - u_c1[1])) / (len(c) - 1)
#
#     return c_convarience
#
# def bias_estimator_convarience(c):
#     c_1 = np.array(c[:, 0])
#     c_2 = np.array(c[:, 1])
#     cov_c1_11_b = np.sum((c_1 - u_c2[0]) * (c_1 - u_c2[0]) / ((len(c))))
#     cov_c1_22_b = np.sum(np.power(c_2 - u_c2[1], 2)) / (len(c))
#     cov_c1_12_b = np.sum((c_1 - u_c2[0]) * (c_2 - u_c2[1])) / (len(c))
#     con_b = np.array([[cov_c1_11_b,cov_c1_12_b],[cov_c1_12_b,cov_c1_22_b]])
#     return con_b
#
#
# print("c1__unbias_convarience = ", unbias_estimator_convarience(c1))
# print("c2__unbias_convarience = ", unbias_estimator_convarience(c2))
#
# print("c1__bias_convarience = ", bias_estimator_convarience(c1))
# print("c2__bias_convarience = ", bias_estimator_convarience(c2))
#
# #=====================Problem 2.2(d) plot========================#
#
# # fig = plt.figure()
# # ax = fig.add_subplot(1,1,1)
# def draw_gaussian_distribution_2_D(u,v,x,y):
#     c1_1 = np.array(c1[:, 0])
#     c1_2 = np.array(c1[:, 1])
#
#     c2_1 = np.array(c2[:, 0])
#     c2_2 = np.array(c2[:, 1])
#
#     # draw scatter plot
#     plt.scatter(c1_1,c1_2)
#     plt.scatter(c2_1,c2_2)
#
#     #produce x,y 2D plane
#     [X, Y] = np.meshgrid(x, y)
#
#     # variance of x
#     sigma_x_quadrat = v[0][0]
#
#     #standard variance of x
#     sigma_x = np.sqrt(sigma_x_quadrat)
#
#     # variance of y
#     sigma_y_quadrat = v[1][1]
#
#     # standard variance of y
#     sigma_y = np.sqrt(sigma_y_quadrat)
#
#     #convarience of x,y
#     sigma_xy = v[0][1]
#
#     r = sigma_xy / (sigma_x * sigma_y)
#     part1 = 1/(2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - r**2))
#     p1 = -1 / (2 * (1 - r**2))
#     px = np.power((X - u[0]),2)/ sigma_x_quadrat
#     py = np.power((Y - u[1]),2) / sigma_y_quadrat
#     pxy = 2 * r * (X - u[0]) * (Y - u[1]) / (sigma_x * sigma_y)
#     Z = part1 * np.exp(p1 * (px - pxy + py))
#     CS = plt.contour(X, Y, Z)
#     plt.clabel(CS, inline=1, fontsize=10)
#     plt.title('gaussian_distribution_2_D')
#
#     return X,Y,Z
#
#
#
# delta = 0.025
# x = np.arange(-10.0, 10.0, delta)
# y = np.arange(-10.0, 10.0, delta)
# X,Y,p_x_c1 = draw_gaussian_distribution_2_D(u_c1,unbias_estimator_convarience(c1),x,y)
# _,_,p_x_c2 = draw_gaussian_distribution_2_D(u_c2,unbias_estimator_convarience(c2),x,y)
# #plt.show()
#
# #=====================Problem 2.2(e) plot========================#
#
# # Problem 2.2(e)  posterior distribution of each class p(Ci|x) and show the decision boundary.
# # p_c_x = p_x_c * p_c
# p_x = p_x_c1 * p_c1 + p_x_c2 * p_c2
# p_c1_x = p_x_c1 * p_c1/ p_x
# p_c2_x = p_x_c2 * p_c2/ p_x
# # CS1 = plt.contour(X, Y, p_c1_x)
# # plt.clabel(CS1, inline=1, fontsize=10)
# # CS2 = plt.contour(X, Y, p_c2_x)
# # plt.clabel(CS2, inline=1, fontsize=10)
# # plt.title('gaussian_distribution_2_D_  posterior distribution')
#
# f = np.sign( p_x_c1 - p_x_c2)
# plt.contour(X,Y,f)
# plt.show()

#====================Problem 2.3 (a) Histogram===================#
def reader(path):
    with open(path) as file:
        data_train = np.loadtxt(file)
    return data_train
train_path = "/Users/apple/Documents/tu_darmstadt/MSC_2semester/statistic_mahine_learning/homework/hw2/dataSets/nonParamTrain.txt"
test_path = "/Users/apple/Documents/tu_darmstadt/MSC_2semester/statistic_mahine_learning/homework/hw2/dataSets/nonParamTest.txt"

data_train = reader(train_path)
data_test = reader(test_path)

number_train_Data = len(data_train)
minimum_data = np.min(data_train)
maxmum_data = np.max(data_train)



min_his = np.floor(minimum_data)
max_his = np.ceil(maxmum_data)

# bins1 = np.ceil((maxmum_data - minimum_data)/0.02)
# bins2 = np.ceil((maxmum_data - minimum_data)/0.5)
# bins3 = np.ceil((maxmum_data - minimum_data)/2)
# fig = plt.figure()
# fig.add_subplot(1,3,1)
# plt.hist(data_train, bins1)
# fig.add_subplot(1,3,2)
# plt.hist(data_train, bins2)
# fig.add_subplot(1,3,3)
# plt.hist(data_train, bins3)
# plt.show()

u_train = np.sum(data_train)/number_train_Data


#===================Problem 2.3 (b) Kernel Density Estimate=============#
def gaussian_distribution(x,u,sigma):
    part_1 = 1/(np.sqrt(2 * np.pi) * sigma)
    part_2 = - np.power((x - u),2) /(2*np.power(sigma,2))
    probability = part_1 * np.exp(part_2)
    return probability


def total_gaussian_distribution(x,sigma,data_set):
    p_x = 0
    for data in data_set:
        p_x += gaussian_distribution(x,data,sigma)
    return p_x

#!!!!!!!!!do not understand!!!!!!!!#
# def gaussian_kernal(x,sigma,data_set):
#     part_1 = 1/(len(data_set) * np.sqrt(2 * np.pi) * sigma)
#     sum = 0
#     for x_n in data_set:
#         part_2 = np.exp( -((x - x_n)**2)/(2 * sigma**2) )
#         sum += part_2
#     p_x = part_1 * part_2
#     return p_x


def calculate_log_likelihoods(sigma,data_set):
    probability_for_data_set = []
    log_likelihod = 0
    for i in data_set:
        p_x = total_gaussian_distribution(i,sigma,data_set)
        p_x = p_x/len(data_set)
        probability_for_data_set.append(p_x)
        if p_x != 0:
            log_p_x = np.log(p_x)
        else:
            log_p_x = 0
        log_likelihod += log_p_x
    return log_likelihod,probability_for_data_set



log_likelihod_1,list_probability_sigma0_03 = calculate_log_likelihoods(0.03,data_train)
log_likelihod_2,list_probability_sigma0_2 = calculate_log_likelihoods(0.2,data_train)
log_likelihod_3,list_probability_sigma0_8 = calculate_log_likelihoods(0.8,data_train)
print("The log-likelihoods are for sigma is 0.03 =" , log_likelihod_1)
print("The log-likelihoods are for sigma is 0.2 =" , log_likelihod_2)
print("The log-likelihoods are for sigma is 0.8 =" , log_likelihod_3)

def show_density_estimates(list_y):
    list = []
    for x,y in zip(data_train,list_y):
        list.append((x,y))
    list.sort()
    x_list = []
    y_list = []
    for x,y in list:
        x_list.append(x)
        y_list.append(y)
    return x_list,y_list

x_list0_03,y_list0_03 = show_density_estimates(list_probability_sigma0_03)
x_list0_2,y_list0_2 = show_density_estimates(list_probability_sigma0_2)
x_list0_8,y_list0_8 = show_density_estimates(list_probability_sigma0_8)
plt.plot(x_list0_03,y_list0_03,"r",
         x_list0_2, y_list0_2,"g",
         x_list0_8, y_list0_8,"b")


#plt.show()


#=====================problem 2_3_c====================#
sorted_data_train = sorted(data_train)

def k_NN(k,data_set):
    list_p_x = []
    log_likelihod = 0
    sum = 0
    np_arr = np.array(data_set)
    np_arr = np_arr.reshape(-1,1)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(np_arr)
    distances, indices = nbrs.kneighbors(np_arr)
    for i in range(len(distances)):
        v = distances[i][k]
        new_p_x = k / (len(distances) * v)
        list_p_x.append(new_p_x)
        sum += new_p_x
        log_p_x = np.log(new_p_x)
        log_likelihod += log_p_x
    # print("distances ===",distances)
    # print("indices ===", indices)
    # print("sum = ",sum)

    return log_likelihod,list_p_x

knn2,list_p_x_2 = k_NN(2,sorted_data_train)
knn8,list_p_x_8 = k_NN(8,sorted_data_train)
knn35,list_p_x_35 = k_NN(35,sorted_data_train)
print("knn2 =" , knn2)
print("knn8 =" , knn8)
print("knn35 =" , knn35)

plt.plot(list(sorted_data_train),list_p_x_2,"r",
         list(sorted_data_train), list_p_x_8,"g",
         list(sorted_data_train), list_p_x_35,"b")
#plt.show()

#========================problem 2.3 (d) Comparison of the Non-Parametric Methods=================#

#====kernal estimator test=====
log_likelihod_test_1,list_probability_sigma0_03 = calculate_log_likelihoods(0.03,data_test)
log_likelihod_test_2,list_probability_sigma0_2 = calculate_log_likelihoods(0.2,data_test)
log_likelihod_test_3,list_probability_sigma0_8 = calculate_log_likelihoods(0.8,data_test)
print("The log-likelihoods are for sigma is 0.03 =" , log_likelihod_test_1)
print("The log-likelihoods are for sigma is 0.2 = " , log_likelihod_test_2)
print("The log-likelihoods are for sigma is 0.8 = " , log_likelihod_test_3)
#====knn test=====
knn2_test,list_p_x_2_test = k_NN(2,data_test)
knn8_test,list_p_x_8_test = k_NN(8,data_test)
knn35_test,list_p_x_35_test = k_NN(35,data_test)
print("knn2_test =" , knn2_test)
print("knn8_test =" , knn8_test)
print("knn35_test =" , knn35_test)



