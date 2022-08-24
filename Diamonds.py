# Myroslava Sánchez Andrade A01730712

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# EXTRACT

print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_EXTRACT_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")

# Imports
print("Importing libraries")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Function that creates a report of the DataFrame (overview, variables, quantile statistics, descriptive statistics, correlations, missing values)
from dataprep.eda import create_report

# GET DATA (https://www.kaggle.com/datasets/shivam2503/diamonds) => Determining the types of each column
types = {
  'carat': float,
  'cut': str,
  'color': str,
  'clarity': str,
  'depth': float,
  'table': float,
  'price': float,
  'x': float,
  'y': float,
  'z': float
}

# Storing the DF in the variable diamonds
print("Reading the original csv from kaggle")
diamonds = pd.read_csv("diamonds.csv", dtype = types)

# Creation of the DF report
print("Creating report")
report = create_report(diamonds, title = 'Diamonds reports')

# Showing the report in the browser (html)
print("Opening report in a browser")
report.show_browser()

# After analyzing and verifying the content of the DF, the variables x, y, z and carat have a correlation > 0.94, which allows use to eliminate 3 and stay with 1. I decided to stay with the carat variable and dropped the others.
print("Dropping the columns x, y and z because of the high correlation with carat")
diamonds = diamonds.drop(columns = ["index", "x", "y", "z"])

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# TRANSFORM

print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_TRANSFORM_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")

# Transformation of percentages values => Reduction of the scale of percentage values from 0-100 to 0-1
def percentages_standardization(df):
    return df / 100

print("Standarizing the percentage variables")
diamonds["depth"] = percentages_standardization(diamonds["depth"])
diamonds["table"] = percentages_standardization(diamonds["table"])


# Standardize dimensions (x, y, z) => Since the columns were dropped the function was never used
def dimensions_standardize(df, max_val):
    return df / max_val


# Function that creates a new column for each category of a categorical variable => The function was never used since the categories could be converted to numerical values
def one_hot_encoder(df, orig_column):
  new_columns = df[orig_column].unique()
  for name in new_columns:
    df[f'{orig_column}_{name}'] = (df[orig_column] == name).astype(int)
  df = df.drop(columns=[orig_column])
  return df


# Function that converts categorical variables to numerical variables => This functon can only be used when the categories can be represented with numerical values => worst-best
def cat_to_scale(df, orig_column, scale):
  scale_dict = {}
  for i in range(len(scale)):
    scale_dict[scale[i]] = i / (len(scale) - 1)
  
  return df.replace({orig_column: scale_dict})

# Variables which categories go from worst to best
print("Converting categorical variables to numerical")
diamonds = cat_to_scale(diamonds, 'cut', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
diamonds = cat_to_scale(diamonds, 'color', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
diamonds = cat_to_scale(diamonds, 'clarity', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', "VVS2", "VVS1", "IF"])

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# LOAD

print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_LOAD_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")

# Creating a new csv with which we will work
print("Exporting a new csv with which we will work")
diamonds.to_csv("clean_diamonds.csv")


# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# MODEL

print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_MODEL (MULTI LINEAR REGRESSION)_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")

# Redefining our DF to store the clean data
print("Reading the cleaned csv")
diamonds = pd.read_csv("clean_diamonds.csv")

max_y_value = diamonds[["price"]].max()

# Df of the variables (inputs)
x = diamonds[["carat", "cut", "color", "clarity", "depth", "table"]]

# Df of the output => We standarize it to be in range 0-1
y = diamonds[["price"]] / max_y_value

cut = int((0.80 * y.count())[0])

# Since the values of the DF were accommodated in descendent order I generated a sample random to train with diverse types of diamonds 
data = x.copy()
data['actual_value'] = y
data = data.sample(frac=1).reset_index(drop=True)
y_sample = data[['actual_value']]
x_sample = data.drop(columns = ['actual_value'])

# Dividing the dfs for the training (80%) and the test(20%)
print("Dividing the data for the training (80%) and the test (20%)")
x_train, x_test = x_sample[0:cut], x_sample[cut:].reset_index(drop = True)
y_train, y_test = y_sample[0:cut], y_sample[cut:].reset_index(drop = True)

# Function that creates random values between -1 and 1 for the weights of the variables and the bias
def init_random_values():
    # Each variable will have a weight
    weights = pd.DataFrame(np.random.random(size = (1, 6))) * 2 - 1
    bias = pd.DataFrame(np.random.random(size = (1, ))) * 2 - 1
    return weights, bias

# Function that makes the prediction given the inputs (DF), the weights and the bias
def get_y_prediction(x, weights,  bias):
    return  ((x.values * weights.values).sum(axis = 1) + bias.values).T

# Total cost function => MSE
def cost(y_pred, y):
    return ((y_pred - y) ** 2).mean()[0]

# Calculating the avg weights slope => derivative of the cost with respect to the weights
def weights_slope(y_pred, y, weights):
    return ((y_pred - y).values * weights.values).mean(axis = 0)

# Calculating the avg bias slope => cost derivative with respect to the bias
def bias_slope(y_pred, y):
    return (y_pred - y).mean()

# Gradient descent function
def gradient_descent(y, x, lr, weights, bias):
    # Prediction of our model
    y_pred = get_y_prediction(x, weights, bias)
    # Slopes of the weights
    weights_delta = weights_slope(y_pred, y, weights)
    # Slope of the bias
    bias_delta = bias_slope(y_pred, y)
    # New weights
    weights = weights - lr * weights_delta
    # New bias
    bias = bias - lr * bias_delta.values
    return weights, bias

# Training
def train(y, x, lr, weights, bias, epochs):
    # Defining a local_error to replace the weights and the bias only if the error is less than the older iteration
    local_error = np.inf
    # Reducing our model error by redefining the weights and the bias each epoch
    for i in range(epochs):
        weights_local, bias_local = gradient_descent(y, x, lr, weights, bias)
        # Calculating the new error
        error = cost(get_y_prediction(x, weights, bias), y)
        # print(error)
        if np.isnan(error) or np.isinf(error) or error < 0.0001 : 
            break
        if error < local_error:
            local_error = error
        else:
            break
        weights, bias = weights_local, bias_local
    return weights, bias

# Defining the learning rate and epochs
learning_rate = 0.05
epochs = 1000

# Creating the random values for the weights and the bias
min_weights, min_bias = init_random_values()
min_cost = cost(get_y_prediction(x_test, min_weights, min_bias), y_test)

weights, bias = min_weights, min_bias

# Defining the maximum error we accept and the max number of iterations
error_max = 0.008
max_it = 400
i = 0

# Calculating the values of the weights and the bias (with the training model), until the error is less than 0.8% or has reached the max iterations
print("Training the model and returning the best weights and bias")
while (curr_cost := cost(get_y_prediction(x_train, weights, bias), y_train)) > error_max and i < max_it:
    # Reasigning random values for the weights and the bias
    weights, bias = init_random_values()
    i+=1
    # Setting the new values of the "betas" after training
    weights, bias = train(y_train, x_train, learning_rate, weights, bias, epochs)
    # If the error is smaller than the last's iteration, we update our minimum variables
    if curr_cost < min_cost:
        min_weights = weights
        min_bias = bias
        min_cost = curr_cost
    print(f'iter {i}/{max_it}. Minimum error until now: {min_cost}')


if min_cost > curr_cost:
    min_cost = curr_cost


# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# RESULTS

print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_RESULTS_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")

# Weights and values after training
print(f'\nBest weights: {min_weights}, best bias: {min_bias}')
print(f'Minimun error after training: {min_cost}')
      
predicted_y = get_y_prediction(x_test, weights, bias)
total_cost = cost(predicted_y, y_test)

print(f"Error of the test: {total_cost}")
results = y_test
results["predicted value"] = predicted_y
results = results * max_y_value[0]

print(results)