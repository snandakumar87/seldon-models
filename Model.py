import joblib
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('ceh-model')

class Model(object):

    def __init__(self):
        print("Initializing.")
        print("Loading model.")
        self.model = joblib.load('model.pkl')

    def predict(self, X, names, meta):
        logger.debug(X)
        _X = pd.DataFrame(X, columns=['age', 'income', 'response', 'events'])
        return self.model.predict_proba(_X)