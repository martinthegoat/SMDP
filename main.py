# coding:utf-8
from MyThread import *

if __name__ == '__main__':
    '''
        Feature Engineering
    '''
    dp = DataPreprocessing()
    X_std, y_std, y_mu, y_sigma, X_train, y_train, X_test, y_test = dp.data_preparation()
    # feature importance analysis
    f_importance = dp.feature_importance_analysis(X_std, y_std)

    '''
    Thread1: Multi-Linear Model
    '''
    # train the model
    model_name = 'multi_linear'
    params1 = {
        'alpha': [0.1, 0.15, 0.3, 0.5]
    }
    thread1 = MyThread(1, 'train and test for multi-linear model',
                       dp, y_mu, y_sigma, model_name, X_train, y_train, X_test, y_test,
                       is_gs=False, params=params1)

    '''
    Thread 2: Ensemble Model -- XGBoost
    '''
    # train the model
    model_name = 'xgboost'
    params2 = {
        'max_depth': [20, 30, 40],
        'n_estimator': [200, 300, 400],
        'reg_lambda': [10, 15, 20],
        'lr': [0.01, 0.1]
    }
    thread2 = MyThread(2, 'train and test for XGBoost model',
                       dp, y_mu, y_sigma, model_name, X_train, y_train, X_test, y_test,
                       is_gs=True, params=params2)

    # start the threads
    thread1.start()
    # thread2.start()
    thread1.join()
    # thread2.join()
    print('exit the main threading')

