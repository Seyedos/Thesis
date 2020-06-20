# Import all the necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy import optimize
from random import shuffle
import numpy as np
import pandas as pd


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


def extract_summary_statistics(X, crossterms_SS):
    # Summary: this function extracts summary statistics from a data set
    # Input: X; the data set
    #        crossterms_SS; a boolean whether or not to add crossterms
    # Output: SS; a list with all calculated summary statistics

    data = np.array(X)
    data_trans = data.transpose()

    logk = data[0, :]
    logr = data[1, :]
    logc = data[2, :]
    logy = data[3, :]

    # Calculate correlation and crosscorrelation
    df = pd.DataFrame(data_trans)
    corr = np.corrcoef(data)
    crosscorr1 = np.correlate(logk[0:3], logr[0:3], "full")
    crosscorr2 = np.correlate(logk[0:3], logc[0:3], "full")
    crosscorr3 = np.correlate(logk[0:3], logy[0:3], "full")
    crosscorr4 = np.correlate(logr[0:3], logc[0:3], "full")
    crosscorr5 = np.correlate(logr[0:3], logy[0:3], "full")
    crosscorr6 = np.correlate(logc[0:3], logy[0:3], "full")

    # Add calculated summary statistics to a list
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

    # Add crossterms if requested
    if crossterms_SS:
        loops = len(SS)
        for i in range(loops):
            for j in range(i, loops):
                SS.append(SS[i]*SS[j])

    return SS


def split_data(X, y, num_sim):
    # Summary: this function splits the data in training and testing sets, in 5 folds for cross validation
    # Input: X; data set
    #        y; parameter settings used to simulate data set
    #        num_sim; number of simulations
    # Output: X_train_list; list with 5 folds of training data 
    #         y_train_list; list with 5 folds of training validating data
    #         X_test_list; list with 5 folds of testing data 
    #         y_test_list; list with 5 folds of testing validating data 

    # Set random state to 1 for reproducable results
    kfold = KFold(5, shuffle=True, random_state=1)
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    for train_index, test_index in kfold.split(X):
        X_train_list.append(X[train_index])
        y_train_list.append(y[train_index])
        X_test_list.append(X[test_index])
        y_test_list.append(y[test_index])

    return X_train_list, y_train_list, X_test_list, y_test_list


def accuracy_visual(predictions, true_values, num_sim, split):
    # Summary: this function plots the predictions and real values of a data set
    # Input: predictions; parameter predictions
    #        true_values; real parameter values
    #        num_sim; number of simulations
    #        split; number of observations in validating data

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
    x2 = np.linspace(0, 3, num=split)
    x3 = np.linspace(0.3, 0.4, num=split)
    x4 = np.linspace(0.9, 1.0, num=split)
    x5 = np.linspace(0.0, 0.1, num=split)
    x6 = np.linspace(0, 3, num=split)

    ax[0, 0].scatter(x1, rho_pred, c='black')
    ax[0, 1].scatter(x2, s2_pred, c='black')
    ax[0, 2].scatter(x3, alpha_pred, c='black')
    ax[1, 0].scatter(x4, beta_pred, c='black')
    ax[1, 1].scatter(x5, delta_pred, c='black')
    ax[1, 2].scatter(x6, nu_pred, c='black')

    ax[0, 0].plot(x1, rho_true, '-')
    ax[0, 1].plot(x2, s2_true, '-')
    ax[0, 2].plot(x3, alpha_true, '-')
    ax[1, 0].plot(x4, beta_true, '-')
    ax[1, 1].plot(x5, delta_true, '-')
    ax[1, 2].plot(x6, nu_true, '-')

    ax[0, 0].set_title('rho')
    ax[0, 1].set_title('s2')
    ax[0, 2].set_title('alpha')
    ax[1, 0].set_title('beta')
    ax[1, 1].set_title('delta')
    ax[1, 2].set_title('nu')

    fig.tight_layout()
    plt.show()


def accuracy_statistics(predictions, true_values, split):
    # Summary: this function calculates accuracy statistics of predictions
    # Input: predictions; parameter predictions
    #        true_values; real parameter values
    #        split; number of observations in validating data

    # Calculate mean of parameters
    aggregate_predictions = np.sum(predictions, axis=0, dtype='float64')
    mean = np.divide(aggregate_predictions, split)

    # Create lists
    bias_list = []
    bias_to_mean_list = []

    # Fill lists
    for i in range(split):
        bias_list.append(true_values[i] - predictions[i])
        bias_to_mean_list.append(true_values[i] - mean)

    # Convert list to array
    bias = np.array(bias_list)
    bias_to_mean = np.array(bias_to_mean_list)

    # Apply transformations to bias
    aggregate_bias = np.sum(bias, axis=0, dtype='float64')
    squared_bias = np.square(bias)
    aggregate_squared_bias = np.sum(squared_bias, axis=0, dtype='float64')
    squared_bias_to_mean = np.square(bias_to_mean)
    aggregate_squared_bias_to_mean = np.sum(
        squared_bias_to_mean, axis=0, dtype='float64')

    # Apply final calculations
    average_bias = np.divide(aggregate_bias, split)
    average_squared_bias = np.divide(aggregate_squared_bias, split)
    RMSE = np.sqrt(average_squared_bias)
    predictivity = (aggregate_squared_bias_to_mean -
                    aggregate_squared_bias)/aggregate_squared_bias_to_mean

    print("\nAverage bias: ", average_bias)
    print("\nRMSE: ", RMSE)
    print("\nPredictivity: ", predictivity)


def neural_network(X_train, y_train, X_test, num_SS):
    # Summary: this function trains a neural network and returns it's predictions
    # Input: X_train; list with 5 folds of training data 
    #        y_train; list with 5 folds of training validating data
    #        X_test; list with 5 folds of testing data 
    #        num_SS; number of summary statistics
    # Output: predictions; parameter predictions made by neural network
    
    # Set up neural network
    model = keras.Sequential()
    model.add(layers.Dense(50, input_dim=num_SS, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(6, activation='relu'))

    # Specify settings
    model.compile(loss='mean_squared_error', optimizer='adam')
    es = EarlyStopping(monitor='loss', patience=10)

    # Train neural network
    for i in range(1, len(X_train)):
        model.fit(X_train[i], y_train[i], epochs=10000,
                  batch_size=5, callbacks=[es])

    # Predict on testing data
    predictions = model.predict(X_test[0])

    return predictions


def elastic_net(X_train, y_train, X_test):
    # Summary: this function trains a elastic net and returns it's predictions
    # Input: X_train; training data 
    #        y_train; validating data
    #        X_test; testing data 
    # Output: predictions; parameter predictions made by elastic net

    # Set up elastic net and initialize list
    predictions = []
    regr = ElasticNet(max_iter=10000)

    # Train model and predict on testing data
    for i in range(6):
        regr.fit(X_train, y_train[:, i])
        predictions.append(regr.predict(X_test))

    return predictions


def main():
    # Choose number of data points you want to simulate
    T = 50
    # Set seed
    np.random.seed(1337)
    # Define lists
    par = []
    par_list = []
    data = []
    SS_list = []

    # Set number of simulations
    power = 4
    num_sim = 10**power

    # Set whether to calculate squares and crossterms of summary statistics
    # Set to 'True' when fitting elastic net and 'False' when fitting neural network
    crossterms_SS = True

    for i in range(0, num_sim):
        # Choose a parameter vector you want to simulate data for
        par.append({"rho": 0+i*10**(-power), "s2": 0+3*i*10**(-power), "alpha": 0.3 +
                    i*10**(-power-1), "beta": 0.9+i*10**(-power-1), "delta": 0+i*10**(-power-1), "nu": 0+3*i*10**(-power)})

        # Call function for simulation
        data.append(SimulateData(par[i], T))

        # Compute summary statistics
        if i > 0:
            SS_list.append(extract_summary_statistics(data[i], crossterms_SS))
            par_list.append(list(par[i].values()))

    # Convert list to array
    X = np.array(SS_list)
    y = np.array(par_list)

    # Split data in training and testing data
    X_train, y_train, X_test, y_test = split_data(X, y, num_sim)
    num_SS = np.size(X, 1)

    predictions_nn = neural_network(X_train, y_train, X_test, y_test, num_SS)
    # predictions_el = elastic_net(X_train[0], y_train[0], X_test[0], y_test[0])

    accuracy_statistics(predictions_nn, y_test[0], np.size(y_test[0], axis=0))
    # accuracy_statistics(np.array(predictions_el).T, y_test[0], np.size(y_test[0], axis=0))

    accuracy_visual(predictions_nn, y_test[0], num_sim, np.size(y_test[0], axis=0))
    # accuracy_visual(np.array(predictions_el).T, y_test[0], num_sim, np.size(y_test[0], axis=0))


if __name__ == "__main__":
    main()
