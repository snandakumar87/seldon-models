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
        _X = pd.DataFrame(X, columns=['income', 'response', 'events'])
        return self.model.predict_proba(_X)

    def metrics(self):
        return [
            # a counter which will increase by the given value
            {"type":"COUNTER","key":"mycounter","value":1}, # a counter which will increase by the given value
            {"type":"GAUGE","key":"mygauge","value":100}, # a gauge which will be set to given value
            {"type":"TIMER","key":"mytimer","value":20.2}, # a timer which will add sum and count metrics - assumed millisecs
            ]