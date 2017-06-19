import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal

np.random.seed(1)

#=============reader======================#

def reader(path):
    with open(path) as file:
        data_train = np.loadtxt(file)
    return data_train
path = "/Users/apple/Documents/tu_darmstadt/MSC_2semester/statistic_mahine_learning/homework/hw2/dataSets/gmm.txt"
data_sets = reader(path)

# #=============initial 4 kinds of labels===============
#
# def initial_4_type_labels():
#     labels = np.random.randint(1,5,len(data_sets))
#     return labels

#=============data preprocessing===============

data = {'x': data_sets[:,0], 'y':data_sets[:,1],'label':np.random.randint(1,5,len(data_sets))}
df = pd.DataFrame(data=data)     # produce a table
#print("======\n", df.head())


fig = plt.figure()
#plt.scatter(data["x"],data['y'],c = data['label'])

#==================Expectation-maximization====================#

# step 1 : inttial guess

guess = {
    'u1' : [-1,-1],
    'sigma1':[[3,0],[0,3]],
    'u2': [0.5, 0.5],
    'sigma2': [[3, 0], [0, 3]],
    'u3': [1.5, 1.5],
    'sigma3': [[1, 0], [0, 1]],
    'u4': [2.5, 2.5],
    'sigma4': [[1, 0], [0, 1]],

    'lambda' : [0.25,0.25,0.25,0.25]   # lambda = p(j|xn)  the probability the point xn belong to j class
}
# guess = {
#     'u1' : [-3,-3],
#     'sigma1':[[2,0],[0,2]],
#     'u2': [1.0, 1.0],
#     'sigma2': [[2, 0], [0, 2]],
#     'u3': [2, 2],
#     'sigma3': [[2, 0], [0, 2]],
#     'u4': [4, 4],
#     'sigma4': [[3, 0], [0, 3]],
#
#     'lambda' : [0.25,0.25,0.25,0.25]   # lambda = p(j|xn)  the probability the point xn belong to j class
# }

# probability that a point came from a Guassian with given parameters
# note that the covariance must be diagonal for this to work
# p(j|xn) = p(xn|j) * p(j)
def prob(sample, u, sig, lambd):
    p = lambd
    for i in range(len(sample)):
        p *= norm.pdf(sample[i], u[i], sig[i][i])
    return p

# step 2(E step) : compute the posterior distribution for each mixture component and for all data point
#          assign every data point to its most likely cluster

def expectation(dataFram, paramaters):
    # dataFram.shape(0) : # rows of this table   shape(1) # col
    for i in range(dataFram.shape[0]):
        list_pro_cluster = []
        x = dataFram["x"][i]
        y = dataFram["y"][i]
        probability_cluster1 = prob([x, y], list(paramaters["u1"]), list(paramaters["sigma1"]), paramaters["lambda"][0])
        probability_cluster2 = prob([x, y], list(paramaters["u2"]), list(paramaters["sigma2"]), paramaters["lambda"][1])
        probability_cluster3 = prob([x, y], list(paramaters["u3"]), list(paramaters["sigma3"]), paramaters["lambda"][2])
        probability_cluster4 = prob([x, y], list(paramaters["u4"]), list(paramaters["sigma4"]), paramaters["lambda"][3])
        list_pro_cluster.append(probability_cluster1)
        list_pro_cluster.append(probability_cluster2)
        list_pro_cluster.append(probability_cluster3)
        list_pro_cluster.append(probability_cluster4)

        most_likely_cluster = list_pro_cluster.index(max(list_pro_cluster)) + 1
        dataFram['label'][i] = most_likely_cluster

    return dataFram

# step 3(M step) : update parameters (lambda, u, sigma)

def maximazation(dataFrame, parameters):
    points_assigned_to_cluster1 = dataFrame[dataFrame['label'] == 1]
    points_assigned_to_cluster2 = dataFrame[dataFrame['label'] == 2]
    points_assigned_to_cluster3 = dataFrame[dataFrame['label'] == 3]
    points_assigned_to_cluster4 = dataFrame[dataFrame['label'] == 4]

    percent_assign_to_cluster1 = len(points_assigned_to_cluster1)/float(len(dataFrame))
    percent_assign_to_cluster2 = len(points_assigned_to_cluster2)/float(len(dataFrame))
    percent_assign_to_cluster3 = len(points_assigned_to_cluster3)/float(len(dataFrame))
    percent_assign_to_cluster4 = len(points_assigned_to_cluster4)/float(len(dataFrame))

    parameters['lambda'] = [percent_assign_to_cluster1,percent_assign_to_cluster2,
                            percent_assign_to_cluster3,percent_assign_to_cluster4]

    parameters['u1'] = [points_assigned_to_cluster1['x'].mean(),
                        points_assigned_to_cluster1['y'].mean()]
    parameters['u2'] = [points_assigned_to_cluster2['x'].mean(),
                        points_assigned_to_cluster2['y'].mean()]
    parameters['u3'] = [points_assigned_to_cluster3['x'].mean(),
                        points_assigned_to_cluster3['y'].mean()]
    parameters['u4'] = [points_assigned_to_cluster4['x'].mean(),
                        points_assigned_to_cluster4['y'].mean()]

    parameters['sigma1'] = [[points_assigned_to_cluster1['x'].std(), 0],
                             [0, points_assigned_to_cluster1['y'].std()]]
    parameters['sigma2'] = [[points_assigned_to_cluster2['x'].std(), 0],
                             [0, points_assigned_to_cluster2['y'].std()]]
    parameters['sigma3'] = [[points_assigned_to_cluster3['x'].std(), 0],
                             [0, points_assigned_to_cluster3['y'].std()]]
    parameters['sigma4'] = [[points_assigned_to_cluster4['x'].std(), 0],
                             [0, points_assigned_to_cluster4['y'].std()]]
    return parameters

# get the distance between points
# used for determining if params have converged

def distance(old_params, new_params):
    dist = 0
    for param in ['u1', 'u2','u3','u4']:
        for i in range(2):
            dist += (old_params[param][i] - new_params[param][i])**2
    return dist * 0.5

#====================== total model ======================#
def p_x_with_total_model(dataFrame, dic_parameters):
    list_p_x = []
    for i in range(len(dataFrame)):
        x = dataFrame['x'][i]
        y = dataFrame['y'][i]
        dataPoint = np.array([x,y])
        model1 = multivariate_normal.pdf(dataPoint,np.array(dic_parameters['u1']),np.array(dic_parameters['sigma1']))
        model2 = multivariate_normal.pdf(dataPoint, np.array(dic_parameters['u2']), np.array(dic_parameters['sigma2']))
        model3 = multivariate_normal.pdf(dataPoint, np.array(dic_parameters['u3']), np.array(dic_parameters['sigma3']))
        model4 = multivariate_normal.pdf(dataPoint, np.array(dic_parameters['u4']), np.array(dic_parameters['sigma4']))
        p_x  = model1 + model2 + model3 + model4
        list_p_x.append(p_x)
    return list_p_x


#====================== log_likelihood ======================#
def log_likelihood(list_p_x):
    log_likilihood = 0
    for p_x in list_p_x:
        log_likilihood += np.log(p_x)
    return log_likilihood
#====================== run ======================#
shift = 5
eps = 0.00001
parameters = guess

i = 0
list_shift = []
df_copy = df.copy()
list_log_likelihood = []
print(df_copy.shape[0])

final_parameters = 0

dataFrame1 = 0
para1 = 0
dataFrame3 = 0
para3 = 0
dataFrame5 = 0
para5 = 0
dataFrame8 = 0
para8 = 0
list_index = []
while shift > eps:
    i += 1
    list_index.append(i)
    # E step:
    updated_labels = expectation(df.copy(), parameters)

    # M step:
    updated_parameters = maximazation(updated_labels, parameters.copy())

    shift = distance(parameters,updated_parameters)
    list_shift.append(shift)

    list_p_x = p_x_with_total_model(updated_labels, updated_parameters)
    _log_likelihood = log_likelihood(list_p_x)
    list_log_likelihood.append(_log_likelihood)
    #print("log_likelihood = ",log_likelihood )

    if(i == 1):
        dataFrame1 = updated_labels
        para1 = updated_parameters
    if(i == 3):
        dataFrame3 = updated_labels
        para3 = updated_parameters
    if(i == 5):
        dataFrame5 = updated_labels
        para5 = updated_parameters
    if (i == 8):
        dataFrame8 = updated_labels
        para8 = updated_parameters


    df = updated_labels
    parameters = updated_parameters
    final_parameters = parameters


print(" shift ",list_shift)

# def prepare_plot(dataFrame, parameters):
#     list_x = dataFrame['x']
#     list_y = dataFrame['y']
#     X,Y = np.meshgrid(list_x,list_y)
#     Z =

def draw_gaussian_distribution_2_D(u,v,x,y):

    # draw scatter plot

    plt.scatter(data["x"], data['y'], c=data['label'])
    #produce x,y 2D plane
    [X, Y] = np.meshgrid(x, y)

    # variance of x
    sigma_x_quadrat = v[0][0]

    #standard variance of x
    sigma_x = np.sqrt(sigma_x_quadrat)

    # variance of y
    sigma_y_quadrat = v[1][1]

    # standard variance of y
    sigma_y = np.sqrt(sigma_y_quadrat)

    #convarience of x,y
    sigma_xy = v[0][1]

    r = sigma_xy / (sigma_x * sigma_y)
    part1 = 1/(2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - r**2))
    p1 = -1 / (2 * (1 - r**2))
    px = np.power((X - u[0]),2)/ sigma_x_quadrat
    py = np.power((Y - u[1]),2) / sigma_y_quadrat
    pxy = 2 * r * (X - u[0]) * (Y - u[1]) / (sigma_x * sigma_y)
    Z = part1 * np.exp(p1 * (px - pxy + py))
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('gaussian_distribution_2_D')

    return X,Y,Z

# fig.add_subplot(2,2,1)
delta = 0.025
x = np.arange(-3.5, 3.5, delta)
y = np.arange(-2.5, 7, delta)
draw_gaussian_distribution_2_D(para5['u1'],para8['sigma1'],x,y)
draw_gaussian_distribution_2_D(para5['u2'],para8['sigma2'],x,y)
draw_gaussian_distribution_2_D(para5['u3'],para8['sigma3'],x,y)
draw_gaussian_distribution_2_D(para5['u4'],para8['sigma4'],x,y)
plt.show()
plt.plot(list_index,list_log_likelihood)


# list_x = dataFrame3['x']
# list_y = dataFrame3['y']
# list_p_x = p_x_with_total_model(dataFrame3, para3)
# fig.add_subplot(2,2,1)
# plt.contour(list_x,list_y,list_p_x)
#
# list_x = dataFrame1['x']
# list_y = dataFrame1['y']
# list_p_x = p_x_with_total_model(dataFrame1, para1)
# fig.add_subplot(2,2,1)
# plt.contour(list_x,list_y,list_p_x)
#
# list_x = dataFrame1['x']
# list_y = dataFrame1['y']
# list_p_x = p_x_with_total_model(dataFrame1, para1)
# fig.add_subplot(2,2,1)
# plt.contour(list_x,list_y,list_p_x)

plt.show()

















