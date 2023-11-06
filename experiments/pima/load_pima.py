import requests
import os
import pandas as pd

def get_dataset():
    if not os.path.exists('pima_dataset.csv'):
        url = 'https://gist.githubusercontent.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f/raw/819b69b5736821ccee93d05b51de0510bea00294/pima-indians-diabetes.csv'
        r = requests.get(url, allow_redirects=True)
        open('pima_dataset.csv', 'wb').write(r.content)

    headers = [
        "Number of times pregnant",
        "Plasma glucose concentration a 2 hours in an oral glucose tolerance test",
        "Diastolic blood pressure (mm Hg)",
        "Triceps skin fold thickness (mm)",
        "2-Hour serum insulin (mu U/ml)",
        "Body mass index (weight in kg/(height in m)^2)",
        "Diabetes pedigree function",
        "Age (years)",
        "Class variable (0 or 1)"
    ]
    dataset = pd.read_csv('pima_dataset.csv', names=headers, skiprows=9)
    return dataset.iloc[:,0:-1].to_numpy(), dataset.iloc[:,-1].to_numpy()