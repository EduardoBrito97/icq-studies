import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.inspection import DecisionBoundaryDisplay


def visualize_prediction(icq,X,y):

    icq.fit(X, y)

    y_pred = icq.predict(X)
    y_pred = np.array(y_pred)

    print(classification_report(y,y_pred))
    print(np.unique(y_pred,return_counts = True))

    plt.plot(X[y_pred == 0][:,0], X[y_pred == 0][:,1],'o')
    plt.plot(X[y_pred == 1][:,0], X[y_pred == 1][:,1],'o')
    plt.show()

def normLinear(V,minimo = None,maximo = None):
    if(not maximo or not minimo):
        X1 = V[:,0]
        X2 = V[:,1]
        
        maximo   = [X1.max(),X2.max()]
        minimo = [X1.min(),X2.min()]

    return [[((v[0]-minimo[0])/(maximo[0]-minimo[0])),((v[1]-minimo[1])/(maximo[1]-minimo[1]))] for v in V]

def showModel(function, dataset_treino, dataset_teste = None, start = (-1,-1),end = (1,1), dataTransform = normLinear, granularity=20):
    
    
    X_, Y_ = np.meshgrid(np.linspace(start[0], end[0], granularity),np.linspace(start[1], end[1], granularity))
    Z = []

    for i in range(granularity):
        temp = [function(x) for x in dataTransform(np.column_stack((X_[i],Y_[i])),start,end)]
        temp2=[]
        for x in temp:
            if(x>1):
                x=1
            if(x<-1):
                x=-1
            temp2.append(x)
        Z.append(temp2)
    print(Z)
    
    Z = np.array(Z)
    # plot
    plt.figure(figsize=(10,8))
    plt.title("Decision Map",fontsize=24)
    
    data = dataset_treino
    if type(data) == tuple:
        X,y = data
    else:
        X,y = data[:,:-1],data[:,-1]
    
    label = 1
    
    # plt.plot([X[j,0] for j in range(len(y)) if y[j]!=label],[X[j,1] for j in range(len(y)) if y[j]!=label],'o',color="blue",markersize=3)
    # plt.plot([X[j,0] for j in range(len(y)) if y[j]==label],[X[j,1] for j in range(len(y)) if y[j]==label],'o',color="red",markersize=3)

    plt.plot(X[:,0][y!=label],X[:,0][y!=label],'o',color="blue",markersize=3)
    plt.plot(X[:,0][y==label],X[:,0][y==label],'o',color="red",markersize=3)

    if dataset_teste is not None:
        
        data = dataset_teste
        if type(data) == tuple:
            X,y = data
        else:
            X,y = data[:,:-1],data[:,-1]
            
        label = 1
        # plt.plot([X[j,0] for j in range(len(y)) if y[j]!=label],[X[j,1] for j in range(len(y)) if y[j]!=label],'x',color="blue",markersize=10)
        # plt.plot([X[j,0] for j in range(len(y)) if y[j]==label],[X[j,1] for j in range(len(y)) if y[j]==label],'x',color="red",markersize=10)

        plt.plot(X[:,0][y!=label],X[:,0][y!=label],'x',color="blue",markersize=10)
        plt.plot(X[:,0][y==label],X[:,0][y==label],'x',color="red",markersize=10)



    levels = np.linspace(-1,1,3)
    cs=plt.contourf(X_, Y_, Z, levels=levels,cmap ='coolwarm',)
    plt.colorbar(cs)
    plt.show()

    return [X_,Y_,Z]


def decision_visualization(clf, X, y):

    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="decision_function",
        alpha=0.9,
        plot_method="contourf",
        cmap = "coolwarm"
    )

    disp.ax_.plot(X[:, 0][y == 0], X[:, 1][y == 0],'o', label = "0", color = "blue")
    disp.ax_.plot(X[:, 0][y == 1], X[:, 1][y == 1],'o', label = "1", color = "red")

    # disp.ax_.set_title("Path length decision boundary \nof IsolationForest")
    # plt.axis("square")
    plt.legend()
    #plt.colorbar(disp.ax_.collections[1])

    plt.show()