# coding:utf-8
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


class DataPreprocessing:
    def __init__(self):
        self.diabetes = None
        self.feature_names = None

    def load_diabetes(self):
        """
        Load diabete information into dictionary

        Returns
        -------
        diabetes : dictionary which contains diabetes feature names, data etc..
        """
        diabetes = datasets.load_diabetes()
        self.diabetes = diabetes
        self.feature_names = diabetes.feature_names
        return diabetes

    def assemble_data(self, diabetes):
        """
        Assemble dataset form as pandas dataframe

        Parameters
        ----------
        diabetes : dictionary which contains diabetes feature names, data etc..

        Returns
        -------
        X : n dimensional array (matrix), shape (n_samples, n_features)
            Features(input varibales).
        y : dependent variable served as target.
        """
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        y = diabetes.target
        return X, y

    def check_outliers(self, X, title, row=2, column=5):
        """
        Check outliers and make plot with boxplot

        Parameters
        ----------
        X : n dimensional array (matrix), shape (n_samples, n_features)
            Features(input varibales).
        title : output boxplot title.
        row : plot layout row number parameter, by default 2 rows
        column : plot layout column number paramter, by default 5 columns.
        """
        file_name = 'figures/' + title + '.png'
        fig, axs = plt.subplots(row, column)
        for index, name in enumerate(X.columns):
            axs[index // column, index % column].boxplot(X[name])
            axs[index // column, index % column].set_title(name)

        fig.tight_layout()
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(top=0.8)
        fig.savefig(file_name)
        # fig.show()

    def remove_outlier(self, X, feature, scale=3):
        """
        Remove outliers beyond a specific Standard Deviation (STD)

        Parameters
        ----------
        X : n dimensional array (matrix), shape (n_samples, n_features)
            Features(input varibales).
        feature : Column name of each independent variables.
        scale : standard deviation value.

        Returns
        -------
        X : Clean independent variables.
        """
        # filter outliers that beyond scale STDs
        X = X[X[feature] < scale * np.std(X[feature])]
        # filter outliers that below scale STDs
        X = X[X[feature] > np.mean(X[feature]) - scale * np.std(X[feature])]

        return X

    def feature_normalize(self, X):
        """
        Normalizes the features(input variables) in X.

        Parameters
        ----------
        X : n dimensional array (matrix), shape (n_samples, n_features)
            Features(input varibale) to be normalized.

        Returns
        -------
        X_norm : n dimensional array (matrix), shape (n_samples, n_features)
            A normalized version of X.
        mu : n dimensional array (matrix), shape (n_features,)
            The mean value.
        sigma : n dimensional array (matrix), shape (n_features,)
            The standard deviation.
        """

        # Note here we need mean of indivdual column here, hence axis = 0
        mu = np.mean(X, axis=0)
        # Notice the parameter ddof (Delta Degrees of Freedom)  value is 1
        sigma = np.std(X, axis=0, ddof=1)  # Standard deviation (can also use range)
        X_norm = (X - mu) / sigma
        return X_norm, mu, sigma

    def target_normalize(self, y):
        """
                Normalizes the target y.

                Parameters
                ----------
                y : one dimensional array (vector), shape (1, n_features)
                    target value to be normalized.

                Returns
                -------
                y_norm : n dimensional array (matrix), shape (n_samples, n_features)
                    A normalized version of X.
                mu : The mean value.
                sigma : The standard deviation.
                """
        mu = np.mean(y)
        sigma = np.std(y, ddof=1)
        y_norm = (y - mu) / sigma
        return y_norm, mu, sigma

    def target_denormalize(self, y_norm, mu, sigma):
        """
                Denormalizes the target.

                Parameters
                ----------
                y_norm : normalized target values, shape (n_samples, ).
                mu: the mean value.
                sigma: the standard deviation.

                Returns
                -------
                y_org : one dimensional array, shape (n_samples, ), denormalized target values.
                """
        y_org = y_norm * sigma + mu
        return y_org

    def feature_importance_analysis(self, X, y):
        """
                Normalizes the features(input variables) in X.

                Parameters
                ----------
                X : n dimensional array (matrix), shape (n_samples, n_features)
                    normalized features.
                y : one dimensional array, shape (n_samples, )
                    normalized target values.

                Returns
                -------
                result : a dict, record the feature names and the corresponding feature importance.
        """
        rf_reg = RandomForestRegressor()
        rf_reg.fit(X, y)
        importances = rf_reg.feature_importances_
        indices = np.argsort(importances)[::-1]
        result = {
            'importance': [],
            'f_name': []
        }
        for i in indices:
            result['importance'].append(importances[i])
            result['f_name'].append(self.feature_names[i])
        print('feature importance:')
        print(result)
        plt.figure(figsize=(12, 8))
        plt.bar(result['f_name'], result['importance'], width=0.6)
        for a, b in zip(result['f_name'], result['importance']):
            plt.text(a, b + 0.01, '%.3f' % b, verticalalignment='top', horizontalalignment='center')
        plt.title('feature importance')
        plt.xlabel('feature names')
        plt.savefig('figures/feature_importance.png')
        return result

    def split_data(self, X, y):
        """
        Split dataset into train set and test set in the ratio of 9:1.

        Parameters
        ----------
        X : n dimensional array (matrix), shape (n_samples, n_features)
            Features(input varibale) to be normalized.
        y : one dimension array, shape (n_samples, 1).
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1)
        return X_train, X_test, y_train, y_test

    def data_preparation(self):
        # load diabetes database
        diabetes = self.load_diabetes()
        X, y = self.assemble_data(diabetes)
        print('Original X shape:', X.shape)

        # EDA outlier detection
        title = 'Different features of oscillations before outlier removal'
        self.check_outliers(X, title)

        # iteratively remove outliers
        for feature in X.columns:
            scale = 2.3
            X = self.remove_outlier(X, feature, scale)

        # filter out valid dependent variables based on features
        y = y[X.index]

        # EDA after outlier removal
        title = 'Different features of oscillations after outlier removal'
        self.check_outliers(X, title)

        # Data normalization (mean=0, std=1)
        X_std, X_mu, X_sigma = self.feature_normalize(X)
        y_std, y_mu, y_sigma = self.target_normalize(y)

        # split train and test set
        X_train, X_test, y_train, y_test = self.split_data(X_std, y_std)
        print('After train test split:')
        print('X_train:', X_train.shape)
        print('y_train:', y_train.shape)
        print('X_test:', X_test.shape)
        print('y_test:', y_test.shape)

        return X_std, y_std, y_mu, y_sigma, X_train, y_train, X_test, y_test
