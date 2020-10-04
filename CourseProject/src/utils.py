import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def prefilter_items(data, take_n_popular=5000, item_features=None):
    """Предфильтрация товаров"""

    # 1. Удаление товаров, со средней ценой <= 1$
    price_to_remove = 1
    less_def_price_item_ids = \
        data.groupby('item_id').filter(
            lambda group: (group['sales_value'].sum() / np.maximum(group['quantity'].sum(), 1)) <= price_to_remove). \
            reset_index()['item_id']
    data = data.loc[~data['item_id'].isin(less_def_price_item_ids)]

    # По популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    data = data.loc[data['item_id'].isin(top)]

    return data
    ###########################################################################################################



    # 2. Удаление товаров со средней ценой > 500
    price_to_remove_2 = 500
    more_def_price_item_ids = \
        data.groupby('item_id').filter(
            lambda group: (group['sales_value'].sum() / np.maximum(group['quantity'].sum(), 1)) > price_to_remove_2). \
            reset_index()['item_id']
    data = data.loc[~data['item_id'].isin(more_def_price_item_ids)]

    # 3. Придумайте свой фильтр
    # Удаление товаров которые купили только 1 раз - непопулярных
    quantity_to_remove = 1
    def_quantity_item_ids = \
        data.groupby('item_id').filter(lambda group: group['quantity'].sum() == quantity_to_remove). \
            reset_index()['item_id']

    data = data.loc[~data['item_id'].isin(def_quantity_item_ids)]

    # 4. Удаление  самых непопулярных товаров (N = take_n_popular)
    not_popular_level = 0.002
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    top_notpopular = popularity[popularity['share_unique_users'] < not_popular_level].item_id.tolist()
    data = data.loc[~data['item_id'].isin(top_notpopular)]


    # 5 Уберем товары, которые не продавались за последние 12 месяцев
    time_from_which_start = 48
    not_sold_from_def_time_item_ids = \
        data[data['week_no'] > (data['week_no'].max() - time_from_which_start)].groupby('item_id'). \
            filter(lambda group: group['quantity'].sum() == 0).reset_index()['item_id']

    data = data.loc[~data['item_id'].isin(not_sold_from_def_time_item_ids)]

    # popularity.sort_values('user_id', ascending=False, inplace=True)
    # top_popular = popularity.head(take_n_popular)['item_id']
    # data = data.loc[data['item_id'].isin(top_popular)]

    # data=data.groupby('item_id').nunique()
    # data.sort_values('user_id', ascending=False, inplace=True)


    # Уберем не интересные для рекоммендаций категории (department)
    min_department_size = 30
    if item_features is not None:
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < min_department_size].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data.loc[~data['item_id'].isin(items_in_rare_departments)]

    # По прибыльности
    # min_sales_value_percent=0.001
    # min_sales_value_item_ids = \
    #    data.groupby('item_id').filter(lambda group: (group['sales_value'].sum()/data['sales_value'].sum()) < min_sales_value_percent). \
    #        reset_index()['item_id']
    # data = data.loc[~data['item_id'].isin(min_sales_value_item_ids)]

    # По популярности (> n покупок в неделю)
    min_quantity = 1
    pop_more_def_quantity_per_week_item_ids = \
        data.groupby('item_id').filter(
            lambda group: (group['quantity'].sum() / data['week_no'].nunique()) < min_quantity). \
            reset_index()['item_id']
    data = data.loc[~data['item_id'].isin(pop_more_def_quantity_per_week_item_ids)]

    # По популярности

    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    data = data.loc[data['item_id'].isin(top)]

    return data


def postfilter_items(recommendations, item_features, user_id, data, item_prices,popular_items, N=5):
    """Пост-фильтрация товаров

    Input
    -----
    recommendations: list
        Ранжированный список item_id для рекомендаций
    item_info: pd.DataFrame
        Датафрейм с информацией о товарах
    """

    # Уникальность
    #     recommendations = list(set(recommendations)) - неверно! так теряется порядок
    unique_recommendations = []
    [unique_recommendations.append(item) for item in recommendations if item not in unique_recommendations]

    # Разные категории
    categories_used = []
    final_recommendations = []

    CATEGORY_NAME = 'sub_commodity_desc'
    for item in unique_recommendations:
        category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]

        if category not in categories_used:
            final_recommendations.append(item)
            unique_recommendations.remove(item)
            categories_used.append(category)



    # 2 новых товара (юзер никогда не покупал)
    min_new_items_count = 2

    user_item_ids = data[data['user_id'] == user_id].groupby('user_id')['item_id'].unique().reset_index()
    user_item_ids = user_item_ids['item_id'][0].tolist()
    user_item_ids_arr = np.array(user_item_ids)

    # Для каждого юзера 5 рекомендаций (иногда модели могут возвращать < 5)
    n_rec = len(final_recommendations)
    if n_rec < N:
        # Более корректно их нужно дополнить топом популярных (например)
        count_add = N - n_rec
        for item in popular_items:
            if item not in final_recommendations and count_add > 0:
                category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]
                if category not in categories_used:
                    final_recommendations.append(item)
                    categories_used.append(category)
                    count_add -= 1
            if count_add == 0:
                break
    else:
        final_recommendations = final_recommendations[:N]

    final_recommendations_arr = np.array(final_recommendations)
    flags = ~np.isin(final_recommendations_arr, user_item_ids_arr)
    needed_new_items_count = min_new_items_count - len(final_recommendations_arr[flags])
    unique_recommendations_arr = np.array(unique_recommendations)
    # print('needed_new_items_count', needed_new_items_count)
    if needed_new_items_count > 0:
        final_recommendations = \
            add_not_bought_items(needed_new_items_count, final_recommendations_arr, \
                                 unique_recommendations_arr, user_item_ids_arr)  # загрузить 2 новых товара

    # 1 дорогой товар, > 7 долларов
    final_recommendations_arr = np.array(final_recommendations)
    needed_expensive_items_count = 1
    price_limit = 7
    final_recommendations = add_expensive_items(min_new_items_count, needed_expensive_items_count, price_limit,
                                                final_recommendations_arr, \
                                                unique_recommendations_arr, user_item_ids_arr, item_prices)

    # Для каждого юзера 5 рекомендаций (иногда модели могут возвращать < 5)
    n_rec = len(final_recommendations)
    if n_rec < N:
        # Более корректно их нужно дополнить топом популярных (например)
        count_add = N - n_rec
        for item in popular_items:
            if item not in final_recommendations and count_add > 0:
                category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]
                if category not in categories_used:
                    final_recommendations.append(item)
                    categories_used.append(category)
                    count_add -= 1
            if count_add == 0:
                break
    else:
        final_recommendations = final_recommendations[:N]

    assert len(final_recommendations) == N, 'Количество рекомендаций != {}'.format(N)
    return final_recommendations


# добавить needed_new_items_count новых товара (юзер никогда не покупал)
def add_not_bought_items(needed_new_items_count, final_recommendations, unique_recommendations, user_item_ids):
    flags_is_in_user_items = np.isin(final_recommendations, user_item_ids)
    indexes_to_remove = []
    for index, item in enumerate(final_recommendations[flags_is_in_user_items]):
        if needed_new_items_count <= index:
            break
        indexes_to_remove.append(np.where(final_recommendations == item))

    final_recommendations_res = np.delete(final_recommendations, indexes_to_remove)

    flags_uniq_is_notin_user_items = ~np.isin(unique_recommendations, user_item_ids)
    unique_recommendations_notin_useritems = unique_recommendations[flags_uniq_is_notin_user_items]
    flags_uniq_is_notin_user_items_recomends = ~np.isin(unique_recommendations_notin_useritems,
                                                        final_recommendations_res)
    final_recommendations_res = final_recommendations_res.tolist()
    unique_recommendations_notin_useritems_recommends = unique_recommendations_notin_useritems[
        flags_uniq_is_notin_user_items_recomends]

   # assert len(unique_recommendations_notin_useritems_recommends) >= needed_new_items_count, \
   #     'Количество уникальных рекомендаций < {}'.format(needed_new_items_count)

    final_recommendations_res.extend(
        unique_recommendations_notin_useritems_recommends[:needed_new_items_count].tolist())

    return final_recommendations_res


def add_expensive_items(min_new_items_count, needed_expensive_items_count, price_limit, final_recommendations,
                        unique_recommendations, user_item_ids, item_prices):
    flags = ~np.isin(final_recommendations, user_item_ids)
    new_items_recommendes = final_recommendations[flags]  # not bought by user
    new_items_flags = np.isin(final_recommendations, new_items_recommendes[:min_new_items_count])
    new_items_recommends = final_recommendations[new_items_flags].tolist()

    indexes_to_remove = []
    for item in new_items_recommends:
        indexes_to_remove.append(np.where(final_recommendations == item))

    final_recommendations = np.delete(final_recommendations, indexes_to_remove)

    for item in new_items_recommends:
        final_recommendations = np.insert(final_recommendations, 0, item)

    curr_needed_expensive_items_count = needed_expensive_items_count

    item_more_price_limit_found = False
    rec_item_prices = item_prices.loc[item_prices['item_id'].isin(final_recommendations)]
    for item in rec_item_prices.values:
        rec_item_id = item[0]
        item_price = item[1]
        if item_price > price_limit and curr_needed_expensive_items_count > 0:
            curr_needed_expensive_items_count -= 1
        if curr_needed_expensive_items_count <= 0:
            item_more_price_limit_found = True
            break

    if not item_more_price_limit_found:
        uniq_item_prices = item_prices.loc[item_prices['item_id'].isin(unique_recommendations)]
        for item in uniq_item_prices.values:
            unique_item_id = item[0]
            item_price = item[1]
            if item_price > price_limit and curr_needed_expensive_items_count > 0:
                if np.all(~np.isin(final_recommendations, unique_item_id)):
                    curr_needed_expensive_items_count -= 1
                    final_recommendations = np.insert(final_recommendations, 0, unique_item_id)
                    final_recommendations = np.delete(final_recommendations, len(final_recommendations) - 1)

            if curr_needed_expensive_items_count <= 0:
                item_more_price_limit_found = True
                break

    if not item_more_price_limit_found:
        item_price_arr = item_prices.loc[item_prices['price'] > price_limit]['item_id'].tolist()
        for item_id in item_price_arr:
            if np.all(~np.isin(final_recommendations, item_id)) and curr_needed_expensive_items_count > 0:
                curr_needed_expensive_items_count -= 1
                final_recommendations = np.insert(final_recommendations, 0, item_id)
                final_recommendations = np.delete(final_recommendations, len(final_recommendations) - 1)
            if curr_needed_expensive_items_count <= 0:
                item_more_price_limit_found = True
                break

    return final_recommendations.tolist()
