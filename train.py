import numpy as np

N = 5000

age = np.random.randint(16, 75, N)

income = age + np.abs(np.random.normal(20, 100, N))

def calculate_class(age, income):
    if age < 30:
        if income >= 300:
            return 2
        elif income >= 200:
            return 1
        else:
            return 0
    elif age < 50:
        if income >= 250:
            return 2
        elif income >= 150:
            return 1
        else:
            return 0
    elif age < 60:
        if income >= 150:
            return 2
        elif income >= 75:
            return 1
        else:
            return 0
    else:
        if income >= 200:
            return 2
        elif income >= 150:
            return 1
        else:
            return 0

# calculate event
def calculate_events(age, income):
    if age < 50:
        p = np.array([0.1, 0.75, 0.6, 0.8, 0.5, 0.79, 0.3])
        p = p / p.sum()
        return np.random.choice(7, 1, p=p)[0]
    else:
        p = np.array([0.2, 0.55, 0.65, 0.7, 0.63, 0.70, 0.4])
        p = p / p.sum()
        return np.random.choice(7, 1, p=p).ravel()[0]

events = list(map(lambda x, y: calculate_events(x, y), age, income))

# AIRLINES = 1
# MERCHANDISE = 2
# HOTEL = 3
# ONLINE_PURCHASE = 4
# UTILITIES = 5
# RESTAURANTS = 6
# OTHERS = 7

def calculate_response(age, event):
    if age < 50:
        if event in [2, 3, 4, 6]:
            return 1 if np.random.random() < 0.6 else 0
        else:
            return 1 if np.random.random() < 0.4 else 0
    else:
        if event in [1, 3, 5, 7]:
            return 1 if np.random.random() < 0.6 else 0
        else:
            return 1 if np.random.random() < 0.4 else 0

_class = list(map(lambda x, y: calculate_class(x, y), age, income))

response = list(map(lambda x, y: calculate_response(x, y), age, events))

def segmentation(_class, response):
    if _class == 2:
        if response == 1:
            return 2
        else:
            return 1
    elif _class == 1:
        if response == 1:
            return 2
        else:
            return 1
    elif _class == 0:
        if response == 1:
            return 1
        else:
            return 0

segment = list(map(lambda x, y: segmentation(x, y), _class, response))

import pandas as pd

data = {'age': age, 'income': income, 'class': _class, 'response': response, 'segment': segment, 'events': events}
df = pd.DataFrame(data)

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

outputs = df['segment']
inputs = df[['age', 'income', 'response', 'events']]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.4, random_state=23)

from sklearn_pandas import DataFrameMapper

def build_RF_pipeline(inputs, outputs, rf=None):
    if not rf:
        rf = RandomForestClassifier()
    pipeline = Pipeline([
        ("mapper", DataFrameMapper([
            (['response', 'events'], preprocessing.OrdinalEncoder()),
            (['age', 'income'], None)
        ])),
        ("classifier", rf)
    ])
    pipeline.fit(inputs, outputs)
    return pipeline

def RF_estimation(inputs, outputs,
                  estimator_steps=10,
                  depth_steps=10,
                  min_samples_split=None,
                  min_samples_leaf=None):
    # hyper-parameter estimation
    n_estimators = [int(x) for x in np.linspace(start=50, stop=100, num=estimator_steps)]
    max_depth = [int(x) for x in np.linspace(3, 10, num=depth_steps)]
    max_depth.append(None)
    if not min_samples_split:
        min_samples_split = [2, 3, 4]
    if not min_samples_leaf:
        min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=random_grid,
                                   n_iter=100, scoring='neg_mean_absolute_error',
                                   cv=3, verbose=1, random_state=42, n_jobs=-1)
    rf_random.fit(inputs, outputs)
    best_random = rf_random.best_estimator_
    print(best_random)
    return best_random

rf = RF_estimation(X_train, y_train, estimator_steps=5, depth_steps=5)
random_forest_pipeline = build_RF_pipeline(X_train, y_train, rf)

rf_predictions = random_forest_pipeline.predict(X_test)
print(f"MSE: {random_forest_pipeline.score(X_test, y_test)*100}%")

import joblib
#save mode in filesystem
joblib.dump(random_forest_pipeline, 'model.pkl')