# coding:utf-8

from xgboost import XGBRegressor
from ModelReg import ModelReg


class EnsembleReg(ModelReg):
    def __init__(self, name):
        super(EnsembleReg, self).__init__(name)
        self.model_name = name
        self.model = None

    def fit(self, X_train, y_train, max_depth=20, n_estimator=300, reg_lambda=10, lr=0.1):
        model = XGBRegressor(objective="reg:squarederror",
                             max_depth=max_depth,
                             n_estimators=n_estimator,
                             reg_lambda=reg_lambda,
                             learning_rate=lr)
        model.fit(X_train, y_train)
        self.model = model

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
