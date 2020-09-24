import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def hit_rate_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)
    hit_rate = (flags.sum() > 0) * 1

    return hit_rate


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list  # Тут нет [:k] !!

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(recommended_list, bought_list)

    if k < len(prices_recommended):
        prices_recommended = prices_recommended[:k]

    prices_recommended_arr = np.array(prices_recommended)

    money_precision = prices_recommended_arr[flags].sum() / prices_recommended_arr.sum()

    return money_precision


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)

    recall = flags.sum() / len(bought_list)

    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(recommended_list, bought_list)

    if k < len(prices_recommended):
        prices_recommended=prices_recommended[:k]

    prices_recommended_arr = np.array(prices_recommended)
    prices_bought_arr = np.array(prices_bought)

    money_precision = prices_recommended_arr[flags].sum() / prices_bought_arr.sum()

    return money_precision


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list  # Тут нет [:k] !!

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision


def map_k(recommended_list, bought_list, users_count, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(recommended_list, bought_list)

    if sum(flags) == 0:
        return 0

    sum_ = 0
    for i in range(1, k + 1):

        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k

    result = sum_ / sum(flags)

    result = result / users_count

    return result


def reciprocal_rank(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    recommended_arr = recommended_list[:k]
    flags = np.isin(recommended_arr, bought_list)

    reverse_ranks = []

    for index, flag in enumerate(flags):
        if flag == True:
            reverse_ranks.append(1 / (index + 1))

    reverse_ranks = np.array(reverse_ranks)

    result = reverse_ranks.sum() / sum(flags)

    return result