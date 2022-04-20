from sklearn.pipeline import Pipeline
from ch3.ml_pipeline.operator import OperatorSpace
from ch3.ml_pipeline.utils import telescope_dataset


class MlPipelineClassifier:

    def __init__(self, pipe_config):

        ops = []
        for _, params in pipe_config.items():
            # operator name
            op_name = params.pop('_name')

            # skips 'none' operator
            if op_name == 'none':
                continue

            op = OperatorSpace.get_operator_by_name(op_name)
            ops.append((op.name, op.clz(**params)))

        self.pipe = Pipeline(ops)

    def train(self, X, y):
        self.pipe.fit(X, y)

    def score(self, X, y):
        return self.pipe.score(X, y)


if __name__ == '__main__':

    pipe_config = {
        'op_1': {
            '_name':      'SelectPercentile',
            'percentile': 2
        },
        'op_2': {
            '_name': 'none'
        },
        'op_3': {
            '_name': 'Normalizer',
            'norm':  'l1'
        },
        'op_4': {
            '_name':          'PCA',
            'svd_solver':     'randomized',
            'iterated_power': 3
        },
        'op_5': {
            '_name':     'DecisionTreeClassifier',
            'criterion': "entropy",
            'max_depth': 8
        }
    }

    model = MlPipelineClassifier(pipe_config)
    X_train, y_train, X_test, y_test = telescope_dataset()
    model.train(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)
