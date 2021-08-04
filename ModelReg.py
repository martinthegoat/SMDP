# coding:utf-8

def write_test_info(test_file: str, test_result: dict):
    with open(test_file, 'w') as f:
        for key in test_result:
            f.write('{}:\t{}\n'.format(key, test_result[key]))
        f.close()


class ModelReg(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        print('model name: ', model_name)

    def get_name(self):
        return self.model_name


