import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def vizualize_prediction(icq,X,y):

    icq.fit(X, y)

    y_pred = icq.predict(X)
    y_pred = np.array(y_pred)

    print(classification_report(y,y_pred))
    print(np.unique(y_pred,return_counts = True))

    plt.plot(X[y_pred == 0][:,0], X[y_pred == 0][:,1],'o')
    plt.plot(X[y_pred == 1][:,0], X[y_pred == 1][:,1],'o')
    plt.show()