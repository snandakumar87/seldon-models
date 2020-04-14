import numpy as np
import pandas as pd

N = 10000

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

df.to_csv(r'data/data.csv')