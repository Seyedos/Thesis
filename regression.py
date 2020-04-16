import random
import numpy as np  
from sklearn.linear_model import LinearRegression
from sklearn import metrics


rng = np.random.RandomState(1)

def linear_model(observations, intercept, b1, b2, b3):
    """
    Consider the following function, which simulates the model: 
    y = intercept + b1*x1 - b2*x2 + b3*x3
    """
    X = 10 * rng.rand(observations, 3)
    y_true = intercept + np.dot(X, [b1, b2, b3]) + rng.randn(observations)
    model = LinearRegression()
    model.fit(X, y_true) 
    y_pred = model.intercept_ + np.dot(X, model.coef_)

    return [y_true, y_pred, model]


def extract_summary_statistics(y_pred, y_true):
    """
    Extract summary statistic from simulated model.
    """
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred) 
    mse = metrics.mean_squared_error(y_true, y_pred) 
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    statistic = [mean_absolute_error, mse, median_absolute_error]

    return statistic

# true structural model
real_intercept = 2
real_b1 = 3
real_b2 = 5
real_b3 = -1
real_y_true = [linear_model(1, real_intercept, real_b1, real_b2, real_b3)[0]]
real_y_pred = [linear_model(1, real_intercept, real_b1, real_b2, real_b3)[1]]
real_statistics = extract_summary_statistics(real_y_pred, real_y_true)
real_input_value = [real_intercept] + real_statistics

# simulations generative model
number_simulations = 10 
number_observations = 10

# training dataset collection
training_data = []

for i in range(0, number_simulations):
    intercept = random.uniform(-5,5)
    b1 = random.uniform(-5,5)
    b2 = random.uniform(-5,5)
    b3 = random.uniform(-5,5)

    y_true = [linear_model(number_observations, intercept, b1, b2, b3)[0]]
    y_pred = [linear_model(number_observations, intercept, b1, b2, b3)[1]]
    sim_statistics = extract_summary_statistics(y_pred, y_true)

    training_data.append([intercept, b1, b2, b3, sim_statistics])

# auxiliary model
beta = []
for i in range(0, number_simulations):
    aux_intercept = training_data[i][0]
    s1 = training_data[i][4][0] 
    s2 = training_data[i][4][1]
    s3 = training_data[i][4][2]

    model_aux = linear_model(number_observations, aux_intercept, s1, s2, s3)[2]
    # beta.append(model_aux.predict(real_input_value))


