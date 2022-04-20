from numpy import arange
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, SelectFwe, VarianceThreshold, SelectFromModel
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    Normalizer, Binarizer, MaxAbsScaler, MinMaxScaler,
    PolynomialFeatures, RobustScaler, StandardScaler,
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Operator:

    def __init__(self, name, clz, params = None):
        """
        Pipeline Operator
        """
        if params is None:
            params = {}
        self.name = name
        self.clz = clz
        self.params = params


class OperatorSpace:
    """
    Registry of all available pipeline operators:
    selectors, transformers, classifiers
    """

    selectors = [
        Operator('SelectFwe', SelectFwe, {
            'alpha': arange(0, 0.05, 0.001).tolist()
        }),

        Operator('SelectPercentile', SelectPercentile, {
            'percentile': list(range(1, 100))
        }),

        Operator('VarianceThreshold', VarianceThreshold, {
            'threshold': [0.0001, 0.001, 0.01, 0.1, 0.2]
        })
    ]

    transformers = [
        Operator('Binarizer', Binarizer, {
            'threshold': arange(0.0, 1.01, 0.05).tolist()
        }),

        Operator('FastICA', FastICA, {
            'tol': arange(0.0, 1.01, 0.05).tolist()
        }),

        Operator('MaxAbsScaler', MaxAbsScaler),

        Operator('MinMaxScaler', MinMaxScaler),

        Operator('Normalizer', Normalizer, {
            'norm': ['l1', 'l2', 'max']
        }),

        Operator('Nystroem', Nystroem, {
            'kernel':       [
                'rbf', 'cosine', 'chi2', 'laplacian', 'polynomial',
                'poly', 'linear', 'additive_chi2', 'sigmoid'
            ],
            'gamma':        arange(0.0, 1.01, 0.05).tolist(),
            'n_components': list(range(1, 11))
        }),

        Operator('PCA', PCA, {
            'iterated_power': list(range(1, 11))
        }),

        Operator('PolynomialFeatures', PolynomialFeatures, {
            'degree': [2, 3, 4]
        }),

        Operator('RBFSampler', RBFSampler, {
            'gamma': arange(0.0, 1.01, 0.05).tolist()
        }),

        Operator('RobustScaler', RobustScaler),

        Operator('StandardScaler', StandardScaler)
    ]

    classifiers = [
        Operator('GaussianNB', GaussianNB),

        Operator('BernoulliNB', BernoulliNB, {
            'alpha': [0.01, 0.1, 1, 10]
        }),

        Operator('MultinomialNB', MultinomialNB, {
            'alpha': [0.01, 0.1, 1, 10]
        }),

        Operator('DecisionTreeClassifier',
                 DecisionTreeClassifier, {
                     'criterion': ["gini", "entropy"],
                     'max_depth': list(range(1, 11))
                 }),

        Operator('RandomForestClassifier',
                 RandomForestClassifier, {
                     'n_estimators': [10, 100],
                     'criterion':    ["gini", "entropy"]
                 }),

        Operator('GradientBoostingClassifier',
                 GradientBoostingClassifier,
                 {
                     'n_estimators':      [100],
                     'learning_rate':     [1e-2, 1e-1, 0.5, 1.],
                     'max_depth':         list(range(1, 11)),
                     'min_samples_split': list(range(2, 21))
                 }),

        Operator('KNeighborsClassifier',
                 KNeighborsClassifier, {
                     'n_neighbors': list(range(1, 101)),
                     'weights':     ["uniform", "distance"],
                     'p':           [1, 2]
                 }),

        Operator('LinearSVC', LinearSVC, {
            'penalty': ["l1", "l2"],
            'loss':    ["hinge", "squared_hinge"],
            'dual':    [True, False],
            'tol':     [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'C':       [1e-3, 1e-2, 1e-1, 1., 10., 25.]
        }),

        Operator('XGBClassifier', XGBClassifier, {
            'n_estimators':     [100],
            'max_depth':        list(range(1, 11)),
            'learning_rate':    [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'subsample':        arange(0.05, 1.01, 0.05).tolist(),
            'min_child_weight': list(range(1, 21)),
        }),

        Operator('SGDClassifier', SGDClassifier, {
            'loss':          ['log', 'hinge', 'perceptron'],
            'penalty':       ['elasticnet'],
            'alpha':         [0.0, 0.01, 0.001],
            'learning_rate': ['invscaling', 'constant'],
            'fit_intercept': [True, False],
            'l1_ratio':      [0.25, 0.0, 1.0, 0.75, 0.5],
            'eta0':          [0.1, 1.0, 0.01],
            'power_t':       [0.5, 0.0, 1.0, 0.1, 100.0, 50.0]
        }),

        Operator('MLPClassifier', MLPClassifier, {
            'alpha':              [1e-4, 1e-3, 1e-2, 1e-1],
            'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.]
        })
    ]

    @classmethod
    def get_operator_by_name(cls, name):
        """
        Returns `Operator` by name
        """
        operators = cls.selectors + cls.transformers + cls.classifiers
        for o in operators:
            if o.name == name:
                return o
        return None
