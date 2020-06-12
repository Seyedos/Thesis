# Import all the necessary packages
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from scipy import optimize
from random import shuffle
import numpy as np
import pandas as pd


# Define all the functions needed


def systemofeq(par, deltas):
    # Summary: this function is needed to compute the solution of the baseline representative agent model
    # Input: deltas; a guess for the solution of the model
    #        par: a dictionary containing the parameters of the baseline model
    # Output: eqs; a 4x1 vector that measures the distance to the solution (the solution needs to satisfy four equations)
    delta_k = deltas[0]
    delta_kz = deltas[1]
    delta_ck = deltas[2]
    delta_cz = deltas[3]
    barr = (1-par.get("beta"))/par.get("beta")
    bark = ((1-par.get("beta")+par.get("delta")*par.get("beta")) /
            (par.get("beta")*par.get("alpha")))**(1/(par.get("alpha")-1))
    barw = (1-par.get("alpha"))*(bark**par.get("alpha"))

    eqs = np.zeros(4)
    eqs[0] = bark*delta_k - bark / \
        par.get("beta") + barw*delta_ck + bark*delta_ck/par.get("beta")
    eqs[1] = bark**par.get("alpha") - barw*delta_cz - \
        bark*delta_cz/par.get("beta") - bark*delta_kz
    eqs[2] = par.get("beta")*(1-par.get("alpha"))*(barr+par.get("delta")) * \
        delta_k + par.get("nu")*delta_ck*delta_k-par.get("nu")*delta_ck
    eqs[3] = par.get("nu")*delta_ck*delta_kz + par.get("nu")*delta_cz*par.get("rho") + par.get("nu")*delta_cz + par.get(
        "beta")*(barr+par.get("delta"))*par.get("rho") - par.get("beta")*(1-par.get("alpha"))*(barr+par.get("delta"))*delta_kz
    return eqs


def SimulateData(par, T):
    # Summary: this function simulates data from the baseline representative agent model
    # Input: par; a dictionary containing the parameters of the baseline model
    #        T; the number of datapoints you want to simulate
    # Output: logk, logr, logc, logy:   Tx1 vectors with data simulated from the model
    #         log k is the natural logarithm of capital, log r of interest rates, log c of consumption, log y of output

    # First: solve the model
    deltainit = 0.01*np.ones(4)
    def fun(x): return systemofeq(par, x)
    deltasol = optimize.root(fun, deltainit)
    deltassol = deltasol.x
    delta_k = deltassol[0]
    delta_kz = deltassol[1]
    delta_ck = deltassol[2]
    delta_cz = deltassol[3]
    # for the model solution, simulate data
    rbar = (1-par.get("beta"))/par.get("beta")
    kbar = ((1-par.get("beta")+par.get("delta")*par.get("beta")) /
            (par.get("beta")*par.get("alpha")))**(1/(par.get("alpha")-1))
    wbar = (1-par.get("alpha"))*kbar**par.get("alpha")
    ybar = kbar**par.get("alpha")
    cbar = wbar + kbar/par.get("beta")

    eps = np.sqrt(par.get("s2"))*np.random.randn(T, 1)

    z = np.zeros(T)
    k = np.zeros(T)
    c = np.zeros(T)
    y = np.zeros(T)
    r = np.zeros(T)
    for t in range(T):
        if t == 0:
            z[t] = eps[t]
            k[t] = 0
        else:
            z[t] = z[t-1]*par.get("rho") + eps[t]
            k[t] = delta_k*k[t-1]+delta_kz*z[t-1]

        c[t] = delta_ck*k[t]+delta_cz*z[t]
        y[t] = z[t] + par.get("alpha")*k[t]
        r[t] = ((rbar+par.get("delta"))/rbar)*z[t] + \
            (par.get("alpha")-1)*((rbar+par.get("delta"))/rbar)*k[t]

    logk = np.log(kbar) + k
    logc = np.log(cbar) + c
    logy = np.log(ybar) + y
    logr = np.log(rbar) + r
    return logk, logr, logc, logy


def extract_summary_statistics(X):
    # Import and sort data
    data = np.array(X)
    data_trans = data.transpose()

    logk = data[0,:]
    logr = data[1,:]
    logc = data[2,:]
    logy = data[3,:]

    # Create summary statisics
    df = pd.DataFrame(data_trans)
    corr = np.corrcoef(data)
    crosscorr1 = np.correlate(logk[0:3], logr[0:3], "full")
    crosscorr2 = np.correlate(logk[0:3], logc[0:3], "full")
    crosscorr3 = np.correlate(logk[0:3], logy[0:3], "full")
    crosscorr4 = np.correlate(logr[0:3], logc[0:3], "full")
    crosscorr5 = np.correlate(logr[0:3], logy[0:3], "full")
    crosscorr6 = np.correlate(logc[0:3], logy[0:3], "full")

    # logk = (logk - np.mean(logk)) / (np.std(logk) * len(logk))
    # logr = (logr - np.mean(logr)) / (np.std(logr))
    # print(logk)
    # print(logr)
    
    # print(np.correlate(logk,logr,'full'))
    # print(np.corrcoef(logk,logr))

    SS = []
    for i in range(0, 4):
        SS.append(df.loc[:, i].mean())
        SS.append(df.loc[:, i].std())
        for j in range(i, 4):
            SS.append(corr[i][j])
    
    for k in range(len(crosscorr1)):
        SS.append(crosscorr1[k])
        SS.append(crosscorr2[k])
        SS.append(crosscorr3[k])
        SS.append(crosscorr4[k])
        SS.append(crosscorr5[k])
        SS.append(crosscorr6[k])

    return SS


def split_data(X,y, num_sim):

    np.random.shuffle(X)
    np.random.shuffle(y)

    split = int(0.3 * num_sim)

    X_train = X[split:, :]
    y_train = y[split:, :]

    X_test = X[:split, :]
    y_test = y[:split, :]

    return X_train, y_train, X_test, y_test, split


def visual_accuracy(predictions, true_values, num_sim, split):
    rho_pred = np.array(predictions[:, 0])
    s2_pred = np.array(predictions[:, 1])
    alpha_pred = np.array(predictions[:, 2])
    beta_pred = np.array(predictions[:, 3])
    delta_pred = np.array(predictions[:, 4])
    nu_pred = np.array(predictions[:, 5])
    
    rho_true = np.array(true_values[:, 0])
    s2_true = np.array(true_values[:, 1])
    alpha_true = np.array(true_values[:, 2])
    beta_true = np.array(true_values[:, 3])
    delta_true = np.array(true_values[:, 4])
    nu_true = np.array(true_values[:, 5])

    fig, ax = plt.subplots(nrows=2, ncols=3)

    x1 = np.linspace(0, 0.1, num=split)
    x2 = np.linspace(1.0, 10.0, num=split)
    x3 = np.linspace(0.3, 0.4, num=split)
    x4 = np.linspace(0.9, 1.0, num=split)
    x5 = np.linspace(0.0, 0.1, num=split)
    x6 = np.linspace(1, 11, num=split)

    ax[0,0].scatter(x1, np.sort(rho_pred), c='black')
    ax[0,1].scatter(x2, np.sort(s2_pred), c='black')
    ax[0,2].scatter(x3, np.sort(alpha_pred), c='black')
    ax[1,0].scatter(x4, np.sort(beta_pred), c='black')
    ax[1,1].scatter(x5, np.sort(delta_pred), c='black')
    ax[1,2].scatter(x6, np.sort(nu_pred), c='black')

    ax[0,0].plot(x1, np.sort(rho_true), '-')
    ax[0,1].plot(x2, np.sort(s2_true), '-')
    ax[0,2].plot(x3, np.sort(alpha_true), '-')
    ax[1,0].plot(x4, np.sort(beta_true), '-')
    ax[1,1].plot(x5, np.sort(delta_true), '-')
    ax[1,2].plot(x6, np.sort(nu_true), '-')

    fig.tight_layout()
    plt.show()


def neural_network(X_train, y_train, X_test, y_test, num_SS):

    model = Sequential()
    model.add(Dense(100, input_dim=num_SS, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(6, activation='relu'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=5)

    # _, accuracy = model.evaluate(X, y)
    # print('Accuracy: %.2f' % (accuracy*100))

    predictions = model.predict(X_test)

    # print('\nneural network:')
    # print('predicted: %s, \nexpected: %s)' % (predictions, y_test))

    return predictions


def elastic_net(X_train, y_train, X_test, y_test):

    regr = ElasticNet(tol=0.000000001, max_iter = 10000000)
    regr.fit(X_train, y_train)

    predictions = regr.predict(X_test)

    # print('\nelastic net:')
    # print('predicted: %s, \nexpected: %s)' % (predictions, y_test))

    return predictions
    

def main():
    # Choose number of data points you want to simulate
    T = 150
    # Set seed
    np.random.seed(0)
    # Define lists
    par = []
    par_list = []
    data = []
    SS_list = []
    power = 2
    num_sim = 10**power

    for i in range(0, num_sim):
        # Choose a parameter vector you want to simulate data for
        par.append({"rho": 0+i*10**(-power), "s2": 1+i*10**(-power+1), "alpha": 0.3+
                    i*10**(-power-1), "beta": 0.9+i*10**(-power-1), "delta": 0+i*10**(-power-1), "nu": 1+i*10**(-power+1)})

        # Call function for simulation
        data.append(SimulateData(par[i], T))

        # Compute summary statistics
        SS_list.append(extract_summary_statistics(data[i]))
        par_list.append(list(par[i].values()))

    X = np.array(SS_list)
    y = np.array(par_list)

    X_train, y_train, X_test, y_test, split = split_data(X,y, num_sim)
    num_SS = np.size(X, 1)

    predictions_nn = neural_network(X_train, y_train, X_test, y_test, num_SS)
    predictions_el = elastic_net(X_train, y_train, X_test, y_test)
    
    visual_accuracy(predictions_nn, y_test, num_sim, split)
    visual_accuracy(predictions_el, y_test, num_sim, split)


if __name__ == "__main__":
    main()
