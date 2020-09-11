import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def prefilter_items(data, take_n_popular=5000):
    """Предфильтрация товаров"""

    # 1. Удаление товаров, со средней ценой < 1$
    price_to_remove_1 = 1
    less_def_price_item_ids = \
        data.groupby('item_id').filter(lambda group: group['sales_value'].mean() < price_to_remove_1). \
        reset_index()['item_id']
    data = data.loc[~data['item_id'].isin(less_def_price_item_ids)]

    # 2. Удаление товаров со средней ценой > 30$
    price_to_remove_2 = 30
    more_def_price_item_ids = \
        data.groupby('item_id').filter(lambda group: group['sales_value'].mean() > price_to_remove_2). \
        reset_index()['item_id']
    data = data.loc[~data['item_id'].isin(more_def_price_item_ids)]

    # 3. Придумайте свой фильтр
    # Удаление товаров которые купили только 1 раз - непопулярных
    quantity_to_remove = 1
    def_quantity_item_ids = \
        data.groupby('item_id').filter(lambda group: group['quantity'].sum() == quantity_to_remove).\
        reset_index()['item_id']

    data = data.loc[~data['item_id'].isin(def_quantity_item_ids)]

    # 4. Выбор топ-N самых популярных товаров (N = take_n_popular)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity.sort_values('user_id', ascending=False, inplace=True)
    top_popular = popularity.head(take_n_popular)['item_id']
    data = data.loc[data['item_id'].isin(top_popular)]

    # data=data.groupby('item_id').nunique()
    # data.sort_values('user_id', ascending=False, inplace=True)

    return data
