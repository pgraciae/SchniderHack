import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import pickle

from utils import Utils


class ModelHandler(Utils):

    models = {'SVM': svm.SVC, 'KNN': neighbors.KNeighborsClassifier,
              'DT': tree.DecisionTreeClassifier, 'XGB': XGBClassifier}
    hyperparams = {'SVM': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': np.logspace(-2, 10, 3),
    }, 'KNN': {
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'n_neighbors': np.arange(3, 10, 2),
        'p': np.arange(1, 3),
    }, 'DT': {
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'splitter': ['best', 'random'],
    }, 'XGB': {
    	'booster': ['gbtree', 'gblinear'],
        'min_child_weight': [0.5, 1, 3, 5, 10],
        'gamma': [0.5, 1, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.5, 1.0, 2.0],
        'max_depth': [4, 26, 32, 64 ]
    }}

    def __init__(self, X, Y, model: str, **kwargs):

        super().__init__()

        self.n_splits = kwargs['n_splits'] if 'n_splits' in kwargs else 5
        if 'n_splits' in kwargs:
            del kwargs['n_splits']
        self.hyperparam = self.hyperparams[model]
        self.hyperparam.update(kwargs)
        self.model = self.models[model](
            **{x: kwargs[x][0] if type(kwargs[x]) == list else kwargs[x] for x in kwargs})
        self.X = self.do_scaling(self._ensure_dimensionalit(X))
        self.Y = Y
        assert(len(Y.shape) == 1)
        self.grid = self._gen_gridSearch(
            self.model, self.hyperparam, self.n_splits)
        self.grid_flag = False

    def fit(self, with_score=True, with_grid=True):
        if with_grid:
            self.grid_flag = True
            self.grid.fit(self.X, self.Y)
            print(f"[INFO] The best parameters are {self.grid.best_params_}")
            print(f"[INFO] The best score is {self.grid.best_score_:.4f}")
            top_vals = self.top_params(0.95, 1).params.values[0]
            print(f"[INFO] The best parameters according to ci are {top_vals}")
            self.model = self.model.__class__(**top_vals)
            self.model.fit(self.X, self.Y)
        else:
            self.model = self.model.fit(self.X, self.Y)
        if with_score:
            pred = self.predict(self.X)
            print(f"[INFO] Train acc  is : {self._acc(pred, self.Y):.4f}")

    def predict(self, X):
        X = self._ensure_dimensionalit(X)
        return self.model.predict(self.do_scaling(X))

    def available_models(self):
        return self.models.keys()
    
    def save(self, name = False):
        if not name:
            name = str(self.model.__class__).split('.')[-1][:-2]
        pickle.dump(self.model, open(name + '.pickle','wb'))
    
    def load(self,path):
        self.model = pickle.load(open(path,'rb'))

        
