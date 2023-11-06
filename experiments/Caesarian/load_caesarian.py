import requests
import os
import pandas as pd
import zipfile

def get_dataset():
    if not os.path.exists('caesarian.zip'):
        url = 'https://archive.ics.uci.edu/static/public/472/caesarian+section+classification+dataset.zip'
        r = requests.get(url, allow_redirects=True)
        open('caesarian.zip', 'wb').write(r.content)

    if not os.path.exists('caesarian.csv.arff'):
        with zipfile.ZipFile('caesarian.zip', 'r') as zip_ref:
            zip_ref.extractall()

    headers = [
        "Age",
        "Delivery number",
        "Delivery time",
        "Blood of Pressure",
        "Heart Problem",
        "Caesarian",
    ]
    dataset = pd.read_csv('caesarian.csv.arff', names=headers, skiprows=17)
    return dataset.iloc[:,0:-1].to_numpy(), dataset.iloc[:,-1].to_numpy()