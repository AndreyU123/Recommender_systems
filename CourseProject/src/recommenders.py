import pandas as pd
import numpy as np

# Для работы с матрицами
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    _user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, item_features,data_for_price_calc, n_factors=20, regularization=0.001, iterations=15,
                 num_threads=4,weighting=True, use_item_prices=False,model_name='ALS',fit_main_model=True,fit_own_model=True,
                 target_values='sales_value',funcs=np.sum):
        # Топ покупок каждого юзера
        self._item_prices = None
        self._model_name=model_name
        if use_item_prices:
            self._item_prices=self.get_item_prices(data_for_price_calc)
        self.top_purchases = data.groupby(['user_id', 'item_id'])['sales_value'].count().reset_index()
        self.top_purchases.sort_values('sales_value', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]
        self.item_features = item_features

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['sales_value'].count().reset_index()
        self.overall_top_purchases.sort_values('sales_value', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self._user_item_matrix = self.prepare_matrix(data,target_values,funcs)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self._user_item_matrix)

        # Словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ
        self.item_id_to_ctm = self.prepare_to_ctm_dict(self.item_features, self._user_item_matrix)  # your_code

        # Own recommender обучается до взвешивания матрицы
        if fit_own_model==True:
            self.own_recommender = self.fit_own_recommender(self._user_item_matrix)

        matrix_to_fit=self._user_item_matrix
        if weighting:
            matrix_to_fit = bm25_weight(self._user_item_matrix.T).T

        if fit_main_model==True:
            self._model = self.fit(matrix_to_fit, n_factors, regularization, iterations, num_threads,model_name)

        self.sparse_user_item = csr_matrix(self._user_item_matrix).tocsr()


        #self.itemid_to_id[999999]=0

    @property
    def user_item_matrix(self):
        return self._user_item_matrix

    @property
    def model(self):
        return self._model

    @property
    def item_prices(self):
        return self._item_prices

    @staticmethod
    def get_item_prices(data_for_price_calc):
        item_prices = data_for_price_calc.groupby(['item_id']).agg(
            {'sales_value': "sum", 'quantity': "sum"}).reset_index()
        item_prices['price'] = item_prices['sales_value'] / item_prices['quantity']
        item_prices.drop(['sales_value', 'quantity'], axis=1, inplace=True)
        return item_prices

    @staticmethod
    def prepare_to_ctm_dict(item_features, _user_item_matrix):
        itemids = _user_item_matrix.columns.values
        item_features = item_features[item_features['item_id'].isin(itemids)]
        item_ids_ctm = item_features[item_features['brand'] == 'Private'].item_id.unique().tolist()

        matrix_itemids = [0 for i in range(len(itemids))]
        item_id_to_ctm = dict(map(lambda itemid, ctm: (itemid, (itemid in item_ids_ctm) * 1), itemids, matrix_itemids))

        return item_id_to_ctm

    @staticmethod
    def prepare_matrix(data,target_values,funcs):
        _user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values=target_values,  # Можно пробовать другие варианты
                                          aggfunc=funcs,
                                          fill_value=0
                                          )
        _user_item_matrix = _user_item_matrix.astype(float)  # необходимый тип матрицы для implicit
        return _user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=16)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(useritemmatrix, n_factors, regularization, iterations, num_threads,model_name):
        """Обучает ALS"""

        model=None
        if model_name=='ALS':
            model = AlternatingLeastSquares(factors=n_factors,
                                            regularization=regularization,
                                            iterations=iterations,
                                            calculate_training_loss=True,
                                            num_threads=num_threads)
        elif model_name=='BPR':
            model = BayesianPersonalizedRanking(factors=n_factors,
                                                regularization=regularization,
                                                iterations=iterations,
                                                num_threads=num_threads)

        model.fit(csr_matrix(useritemmatrix).T.tocsr(), show_progress=True)

        return model


    def get_similar_items_recommendation(self, user, filter_ctm=True, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        popularity = self.top_purchases[self.top_purchases['user_id'] == user].head(N)
        # СТМ = товары под брендом Private
        if filter_ctm == True:
            ctm = self.item_features[self.item_features['brand'] == 'Private'].item_id.unique()
            popularity = popularity[~popularity['item_id'].isin(ctm)]

        popularity['similar_recommendation'] = popularity['item_id'].apply(lambda x: self.get_rec(x, N))
        res = popularity['similar_recommendation'].values.tolist()
        res = self._extend_with_top_popular(res, N=N)

        return res

    def get_rec(self, itemid, N):
        recs = self._model.similar_items(self.itemid_to_id[itemid], N=N)
        top_rec = recs[1][0]

        return self.id_to_itemid[top_rec]

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        res = []

        # Находим топ-N похожих users
        similar_users = self._model.similar_users(self.userid_to_id[user], N=N + 1)
        similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:]  # удалим юзера из запроса

        for user in similar_users:
            res.extend(self.get_own_recommendations(user, N=1))

            res = self._extend_with_top_popular(res, N=N)

        return res

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})
            return True

        return False

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        is_updated = self._update_dict(user_id=user)
        if not is_updated:
            return self.get_recommendations(user, model=self.own_recommender, N=N)
        else:
            res = []

        res = self._extend_with_top_popular(res, N=N)

        return res

    def get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        is_updated=self._update_dict(user_id=user)
        if not is_updated:
            res=[]
            if self._model_name=='ALS':
                res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                                            user_items=self.sparse_user_item,
                                                                            N=N,
                                                                            filter_already_liked_items=False,
                                                                            filter_items=None, #[self.itemid_to_id[999999]],
                                                                            recalculate_user=True)]
            elif self._model_name=='BPR':
                res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                                            user_items=self.sparse_user_item,
                                                                            N=N,
                                                                            filter_already_liked_items=False,
                                                                            filter_items=None
                                                                            )]
        else:
            res=[]

        res = self._extend_with_top_popular(res, N=N)

        return res

    def get_recomendations_per_user(self,user, model, N=5):
        recs = []
        if self._model_name == 'ALS':
            recs = model.recommend(userid=self.userid_to_id[user],  # userid - id от 0 до N
                                user_items=self.sparse_user_item,  # на вход user-item matrix
                                N=N,  # кол-во рекомендаций
                                filter_already_liked_items=False,
                                filter_items=None,
                                recalculate_user=False)
        elif self._model_name == 'BPR':
            recs = model.recommend(userid=self.userid_to_id[user],
                                   user_items=self.sparse_user_item,
                                   N=N,
                                   filter_already_liked_items=False,
                                   filter_items=None
                                   )
        return recs

    def fill_prices(self,item_ids):
        price_arr = []
        for item_id in item_ids:
            price_arr.append(self._item_prices[self._item_prices['item_id'] == item_id]['price'].values[0])

        return price_arr

