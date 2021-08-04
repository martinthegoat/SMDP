# coding:utf-8

import threading
from DataPreprocessing import *
from MultiLinearReg import *
from EnsembleReg import *
from sklearn.metrics import mean_absolute_error, r2_score


def print_write_test_info(test_file: str, test_result: dict, mode='w'):
    with open(test_file, mode) as f:
        for key in test_result:
            f.write('{}:\t{}\n'.format(key, test_result[key]))
            print('{}:\t{}'.format(key, test_result[key]))
        f.close()


def cal_test_metrics(y_test, y_pred, y_test_norm, y_pred_norm):
    mse = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_norm, y_pred_norm)
    result = {
        'rmse': rmse,
        'r2': r2
    }
    return result


def plot_fitting_result(y_test, y_pred, model_name):
    fig_file = 'figures/' + model_name + '_fitting_result.png'
    plt.figure(figsize=(12, 8))
    index = range(1, len(y_pred) + 1)
    plt.plot(index, y_test, color='blue', label='true')
    plt.scatter(index, y_test, color='#4f86c6')
    plt.plot(index, y_pred, color='#d81159', label='predicted')
    plt.scatter(index, y_pred, color='#c03546')
    plt.legend(loc='best')
    plt.xlabel('number of samples')
    plt.ylabel('values')
    plt.title('fitting result')
    plt.savefig(fig_file)


def model_test(dp, y_mu, y_sigma, model, model_name, X_test, y_test, is_save_result=False):
    y_pred = model.predict(X_test)
    y_pred_org = dp.target_denormalize(y_pred, y_mu, y_sigma)
    y_test_org = dp.target_denormalize(y_test, y_mu, y_sigma)
    model_test_result = cal_test_metrics(y_test_org, y_pred_org, y_test, y_pred)
    if is_save_result:
        model_test_result_file = 'test_results/' + model_name + '_test_result.txt'
        print('test result for {} regression model:'.format(model_name))
        print_write_test_info(model_test_result_file, model_test_result)
    plot_fitting_result(y_test_org, y_pred_org, model_name)
    return model_test_result


class MyThread(threading.Thread):
    def __init__(self, thread_id, name, dp, y_mu, y_sigma, model_name, X_train, y_train, X_test, y_test, is_gs=False,
                 params=None):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.dp = dp
        self.y_mu = y_mu
        self.y_sigma = y_sigma
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.is_gs = is_gs
        self.params = params

    def run(self) -> None:
        print('*' * 4 + 'start thread: {}'.format(self.name) + '*' * 4)
        if self.is_gs and self.params is not None:
            if self.model_name is 'multi_linear':
                # params: alpha [0.1, 0.15, 0.3, 0.5]
                test_file = 'test_results/{}_gs.txt'.format(self.model_name)
                model_num = 1
                r2_list = []
                for alpha in self.params['alpha']:
                    model_name = self.model_name + str(model_num)
                    model = MultiLinear(model_name)
                    model.fit(self.X_train, self.y_train, alpha=alpha)
                    test_result = model_test(self.dp, self.y_mu, self.y_sigma, model, model_name,
                                             self.X_test,
                                             self.y_test,
                                             is_save_result=False)
                    if model_num == 1:
                        mode = 'w'
                    else:
                        mode = 'a'
                    with open(test_file, mode) as f:
                        f.write('\n' + '*' * 4 + str(model_num) + '*' * 4 + '\n')
                        f.write('alpha:\t{}\n'.format(alpha))
                        print('alpha:', alpha)
                        f.close()
                    print_write_test_info(test_file, test_result, mode='a')
                    r2 = test_result['r2']
                    r2_list.append(r2)
                    model_num += 1
                max_model_num = np.argmax(r2_list)
                with open(test_file, 'a') as f:
                    f.write('\n' + '*' * 12 + '\n')
                    f.write('best model:\t{}\n'.format(max_model_num+1))
            else:  # str(self.model_name).startswith('xgboost')
                # params: max_depth, n_estimator, reg_lambda, lr
                model_num = 1
                test_file = 'test_results/{}_gs.txt'.format(self.model_name)
                r2_list = []
                for max_depth in self.params['max_depth']:
                    for n_est in self.params['n_estimator']:
                        for reg_lambda in self.params['reg_lambda']:
                            for lr in self.params['lr']:
                                model_name = self.model_name + str(model_num)
                                model = EnsembleReg(model_name)
                                model.fit(self.X_train, self.y_train, max_depth=max_depth,
                                          n_estimator=n_est, reg_lambda=reg_lambda, lr=lr)
                                test_result = model_test(self.dp, self.y_mu, self.y_sigma, model, model_name,
                                                         self.X_test,
                                                         self.y_test,
                                                         is_save_result=False)
                                if model_num == 1:
                                    mode = 'w'
                                else:
                                    mode = 'a'
                                with open(test_file, mode) as f:
                                    f.write('\n' + '*' * 4 + str(model_num) + '*' * 4 + '\n')
                                    f.write('max_depth:\t{}\n'.format(max_depth))
                                    print('max_depth:\t{}'.format(max_depth))
                                    f.write('n_est:\t{}\n'.format(n_est))
                                    print('n_est:\t{}'.format(n_est))
                                    f.write('reg_lambda:\t{}\n'.format(reg_lambda))
                                    print('reg_lambda:\t{}'.format(reg_lambda))
                                    f.write('lr:\t{}\n'.format(lr))
                                    print('lr:\t{}'.format(lr))
                                    f.close()
                                print_write_test_info(test_file, test_result, mode='a')
                                r2 = test_result['r2']
                                r2_list.append(r2)
                                model_num += 1
                max_model_num = np.argmax(r2_list)
                with open(test_file, 'a') as f:
                    f.write('\n' + '*' * 12 + '\n')
                    f.write('best model:\t{}\n'.format(max_model_num+1))
        else:
            if self.model_name is 'multi_linear':
                model = MultiLinear(self.model_name)
                theta, cost_history = model.fit(self.X_train, self.y_train)
                print('Final value of theta:', theta)
            else:  # 'xgboost'
                model = EnsembleReg(self.model_name)
                model.fit(self.X_train, self.y_train)
            model_test(self.dp, self.y_mu, self.y_sigma, model, self.model_name, self.X_test, self.y_test)
        print('*' * 4 + 'end thread: {}'.format(self.name) + '*' * 4)
