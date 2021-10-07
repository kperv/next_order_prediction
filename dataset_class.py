"""
A class to transform the original order history dataframe
with scattered info into matrices
for Recurrent Model training and next cart prediction.
"""


import numpy as np
import pandas as pd
from torch.utils.data import Dataset


QUANTILE = 0.7 # The part of the full history to be used for training


class CartDataset(Dataset):
    """ Creation and functionality of a history of user orders dataset.

    Attributes:
        orders: a DataFrame of all orders made by users
                collected in user order
        full_carts: a DataFrame with sets of all cart items
                    ever purchased by a user
        users: a list of user indexes as in the original DataFrame
        feature_dum: int, a history length to form feature vectors

    """

    def __init__(self, orders, full_carts, users, feature_dim):
        self.orders = orders
        self.full_carts = full_carts
        self.users = users
        self.feature_dim = feature_dim
        self.train_mode = True

    def get_orders(self):
        return self.orders

    def get_full_carts(self):
        return self.full_carts

    def get_users(self):
        return self.users

    def get_feature_dim(self):
        return self.feature_dim

    def change_mode(self):
        self.train_mode = False

    def _get_data_target(self, df):
        """ The last user order is used as the target for prediction
        all the previous used for training. """
        return df.iloc[:-1], df.iloc[-1]

    def _get_full_categories_list(self, user_index):
        """ List of all categories for the user. """
        return list(self.full_carts.loc[user_index].values[0])

    def _duplicate_matrix(self, matrix):
        """ If the history of orders is smaller than necessary,
        repeat it to fill the matrix. """
        repeat = self.feature_dim // matrix.shape[-1]
        rem = self.feature_dim % matrix.shape[-1]
        matrix = np.repeat(matrix, repeat, axis=1)
        if rem:
            matrix = np.concatenate((matrix[:, :rem], matrix), axis=1)
        return matrix

    def _vectorize_data(self, data, full_cart):
        """ Transform orders to a one-hot matrix
        of the shape (features, possible categories).
        Features are collected from item history of all orders"""

        # check the number of orders for this user
        # and if it is lower than the num of features
        # if it is not lower, take latest num_features orders
        extent = len(data) < self.feature_dim
        if not extent:
            data = data[-self.feature_dim:]

        # create zero matrix and vectorize one-hot orders
        # according to the full possible categories vector
        # for this user
        matrix = np.zeros((len(full_cart), len(data)))
        for i, order in enumerate(data.cart.values):
            for x in order:
                pos = full_cart.index(x)
                matrix[pos][i] = 1

        # if there is not enough orders - repeat existing orders
        # to fill the matrix
        if extent:
            matrix = self._duplicate_matrix(matrix)

        # matrix shapes should be equal in size
        assert matrix.shape == (len(full_cart), self.feature_dim)
        return matrix

    def _vectorize_target(self, target, full_cart):
        """ Transform the target vector of category_ids to one-hot."""
        target_vector = np.zeros((len(full_cart)))
        for item in target.cart:
            pos = full_cart.index(item)
            target_vector[pos] = 1
        return target_vector

    def _vectorize(self, index, mode='train'):
        """ Transform dataframe grouped-by the user_id
        into one-hot data matrix and target vector"""

        user_index = self.users[index]
        full_cart = self._get_full_categories_list(user_index)

        # for prediction, history with the last element and category_ids
        if not self.train_mode:
            data = self.orders.loc[user_index]
            data = self._vectorize_data(data, full_cart)
            category_ids = [";".join((str(user_index), item))
                            for item in map(str, full_cart)]
            return data, category_ids

        # for training history without the latest element
        # and the latest as a target
        data, target = self._get_data_target(
            self.orders.loc[user_index]
        )
        data = self._vectorize_data(data, full_cart)
        target = self._vectorize_target(target, full_cart)
        return data, target

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self._vectorize(index)

    @classmethod
    def from_df(cls, df):
        """ Transform the dataframe for future vectorization
        and initialize a Dataset object. """
        df1 = df.groupby(['user_id']).count()
        users = df1.index.to_numpy()
        feature_dim = int(df1.cart.quantile(QUANTILE))
        full_carts = df.groupby(['user_id']).agg(
            lambda x: set(x.tolist())).drop(columns=['order_completed_at'])
        orders = pd.pivot_table(
            df,
            values=['cart'],
            index=['user_id', 'order_completed_at'],
            aggfunc=lambda x: x.tolist())
        return cls(orders, full_carts, users, feature_dim)