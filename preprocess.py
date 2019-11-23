import numpy as np
import pandas as pd

def normalize_data(data):
    x_mean=[]
    x_std=[]
    x_mean.append(1)
    x_std.append(0)
    for i in range(1,len(data.columns)):
        st = data.columns[i]
        temp = data[st].values
        data.drop([st],axis=1);
        xm = np.mean(temp)
        xs = np.std(temp)
        temp = temp - xm
        temp = temp/xs
        x_mean.append(xm)
        x_std.append(xs)
        data[str(i)]=temp
    return data,x_mean,x_std

def preprocess(path,deg):
    np.random.seed(42)
    data = pd.read_csv(path)
    sdf = data.values
    np.random.shuffle(sdf)
    data = pd.DataFrame(sdf,columns=['0','1','2','3'])
    Y = data['3'].values
    ones = np.ones(len(Y),dtype=float)
    data['0'] = ones
    data = data.drop(['3'],axis=1)
    data,x_mean,x_std = normalize_data(data)
    cnt = 3
    x1 = data['1'].values
    x2 = data['2'].values
    for i in range(0,deg+1):
        for j in range(0,deg+1):
            if i+j>1 and i+j<=deg:
                t1 = np.array([x**i for x in x1])
                t2 = np.array([x**j for x in x2])
                t3 = t1*t2
                data[str(cnt)] = t3
                cnt+=1
    y_mean = np.mean(Y)
    y_std = np.std(Y)
    Y = Y - y_mean
    Y = Y/y_std
    X = data.values
    X_test = X[:][300000:]
    Y_test = Y[300000:]
    X = X[:][:300000]
    Y = Y[:300000]
    return X,Y,x_mean,y_mean,x_std,y_std,X_test,Y_test


# X,Y,x_mean,y_mean,x_std,y_std = preprocess("3D_spatial_network.csv",2)
# print(len(X),len(Y))
# print(x_mean)
# print(x_std)
# print(X[:5],Y[:5])
