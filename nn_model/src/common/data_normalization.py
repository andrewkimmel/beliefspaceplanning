import numpy as np
import scipy.io
import copy
import pickle


def min_max_normalize(data, min_arr, max_arr):
    normalized_data = copy.deepcopy(data)
    for i in range(data.shape[0]):
        normalized_data[i] = (data[i] - min_arr[:data.shape[1]]) / (max_arr[:data.shape[1]] - min_arr[:data.shape[1]])
    return normalized_data


def min_max_denormalize(data, min_arr, max_arr):
    denormalized_data = copy.deepcopy(data)
    for i in range(data.shape[0]):
        denormalized_data[i] = data[i] * (max_arr[:data.shape[1]] - min_arr[:data.shape[1]]) + min_arr[:data.shape[1]]
    return denormalized_data


def z_score_normalize(data, mean_arr, std_arr):
    normalized_data = copy.deepcopy(data)
    for i in range(data.shape[0]):
        normalized_data[i] = (data[i] - mean_arr[:data.shape[1]]) / std_arr[:data.shape[1]]
    return normalized_data


def z_score_denormalize(data, mean_arr, std_arr):
    denormalized_data = copy.deepcopy(data)
    for i in range(data.shape[0]):
        denormalized_data[i] = data[i] * std_arr[:data.shape[1]] + mean_arr[:data.shape[1]]
    return denormalized_data

def clean_data(data):
    data_copy = copy.deepcopy(data)
    delta = data_copy[:, 6:10] - data_copy[:, :4]
    del_list = []
    for i in range(len(delta)):
        if delta[i][0] < -6 or delta[i][0] > 6 or delta[i][1] < -6 or delta[i][1] > 6 or delta[i][2] < -1 or delta[i][2] > 1 or delta[i][3] < -1 or delta[i][3] > 1:
            del_list.append(i)
    clean_data = np.delete(data_copy, del_list, 0)
    return clean_data
