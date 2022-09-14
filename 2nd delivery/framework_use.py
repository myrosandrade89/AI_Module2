#!/usr/bin/env python
# coding: utf-8

# # **House Sales in King County, USA**
# 
# *Author: Myroslava Sánchez Andrade*
# <br>*Date of creation: 08/09/2022*
# <br>*Last updated: 12/09/2022*

# ---
# 
# ## **Overview**
# 
# The purpose of this repository is the programming of a prediction algorithm using a framework. This development was divided in three main steps:
# 
# - ***ETL***
# <br>In this step, I extratected a dataset from **[Kaggle](https://www.kaggle.com/)**. This dataset was analyzed to perform the corresponding transformations and thus have an ordered and clean data series for the model.
# 
# - ***Modeling***
# <br>There were 3 models used for the prediction. After analizyig the performance of each, I chose the one with best performance and discarded the others.
# 
# - ***Model regularization***
# <br>After analyzing the chosen model, I used regularization and parameter adjustment to improve the performance of the model.

# ---
# ## **Extract**
# The dataset used for this project is **[House sales in KC, Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?select=kc_house_data.csv)**, downloaded from the plataform Kaggle.
# <br>This dataset contains the homes sold between May 2014 and May 2015 in King County. It is composed of the next variables:
# - ***Independent variables:*** id (unic identifier for the house), date (date in which a house was put up for sale), bedrooms, bathroomns, sqft_living (living area inside the house), sqft_lot (total area of the property), floors, waterfront (sea view[boolean]), view (score), condition (score), grade (overall score), sqft_above (attic), sqft_basement, yr_built (year in which the house was built), year_renovated (last year of renovation), zipcode, lat (latitude), long (longitude), sqft_living15, sqft_lot15
#     <br>Being categorical the following variables: `['view', 'condition', 'grade']`
# 
# - ***Dependent variable:*** price (price at which a house was sold)

# In[28]:


# REQUIRED LIBRARIES
# !pip install pandas numpy matplotlib dataprep sklearn tensorflow mlxtend


# In[2]:


# RUN ONLY FOR GOOGLE COLAB

# from google.colab import drive

# drive.mount("path")  

# %cd "path"


# In[14]:


# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
# Function that creates a report of the DataFrame (overview, variables, quantile statistics, descriptive statistics, correlations, missing values)
from dataprep.eda import create_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, regularizers, models
from mlxtend.evaluate import bias_variance_decomp

# Make the numpy printouts easier to read
np.set_printoptions(precision=3, suppress=True)


# In[4]:


print('Tensorflow: ', tf.__version__)


# In[5]:


# Reading data via Pandas from CSV
house_data = pd.read_csv('data/kc_house_data.csv')


# #### ***Verifying structure and content***

# In[6]:


# Validating the information of each column => There are no null values in the whole DF
house_data.info()


# In[7]:


# Creation of the DF report and saving to HTML
report = create_report(house_data, title = 'House data report')


# ***DataFrame report***
# <br>In this report we can identify that there are no missing cells, the number of categorical  variables, the correlation between the variables and the ranges of the values of each variable.

# In[15]:


# # Saving the report
report.show()


# ---
# ## **Transform**
# After verifying the structure and content of the DataFrame and analyzing it, we could identify that there are 3 main processes to be done: dropping columns that do not contribute to the model, normalization of the columns, restructuring the date column.

# #### ***Date transformation***

# In[9]:


# Extracting month and year from date column
house_data.loc[:, 'month'] = pd.DatetimeIndex(house_data['date']).month
house_data.loc[:, 'year'] = pd.DatetimeIndex(house_data['date']).year


# #### ***Column selection***
# Dropping the columns lat, long and zipcode because they do not contribute to the enrichment of the dataset since they are location variables; for the relation of price given a location, we can use the variables sqft_living15 and sqft_lot15 that give information of the houses around a specific house. The id was also dropped since we can use the index of the DataFrame and the date was dropped since it was transformed in ceell above.

# In[10]:


# Dropping location-related columns (lat, long, zipcode) and date (since it was processed in the last step)
house_data = house_data.drop(['id', 'lat', 'long', 'zipcode', 'date'], axis=1, errors='ignore')
print(f"Original data shape: {house_data.shape}")


# In[11]:


house_data


# #### ***Splitting into train and test sets***
# Train data => 80%
# <br>Test data => 20%
# <br>(The validation data split will be done in the model training step)

# In[12]:


# Setting 80% of the data as train and the remaining 20% as test. Validation split will be done until the model training step.
train_dataset = house_data.sample(frac=0.8)
test_dataset = house_data.drop(train_dataset.index)


# In[13]:


# Getting features (X)
train_features = train_dataset.copy()
test_features = test_dataset.copy()

# Getting labels (y)
train_labels = train_features.pop('price')
test_labels = test_features.pop('price')


# In[14]:


# Verifying shape of features and labels arrays)
print(f'Features shape (X): {train_features.shape}')
print(f'Labels shape (y): {train_labels.shape}')


# #### ***Normalization***

# In[38]:


# Normalization fitting for train features (X)
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy().shape)


# #### ***Load***
# Exporting the cleaned and transformed DataFrames

# In[ ]:


train_features.to_csv("data/train_features.csv")
train_labels.to_csv("data/train_labels.csv")
test_features.to_csv("data/test_features.csv")
test_labels.to_csv("data/test_labels.csv")


# ---
# ## **Model**
# For the modeling, there were used two models: linear regression and a neural network. The one that fitted the best for the given dataset, would be regularized and adjusted to improve the performance of the model.

# In[15]:


# Defining array to store the predictions of each model
test_results = {}


# In[16]:


# Function for plotting the loss history for each model
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error [Price]')
  plt.legend()
  plt.grid(True)


# #### ***Model: Generic method***

# In[52]:


def model_definition_compilation_fit(name, layers, learning_rate, epochs, validation_split):
    # Model definition and compilation using MAE
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_absolute_error'
    )
    
    # Model fit using training features and labels
    history = model.fit(
        train_features,
        train_labels,
        epochs=epochs,
        # Suppress logging.
        verbose=1,
        # Calculate validation results on % of the training data.
        validation_split = validation_split
    )
    
    # Plotting the loss history for the model
    plt.clf()
    plot_loss(history)
    plt.show()
    
    # Adding the model predictions to the previously defined dictionary
    test_results[name] = model.evaluate(
        test_features, test_labels, verbose=1
    )
    
    # Calculate bias and variance
    mse, bias, var = bias_variance_decomp(model, train_features.values, train_labels.values, test_features.values, test_labels.values, loss='mse', num_rounds=2, epochs=epochs)

    print('MAE from bias_variance lib [avg expected loss]: %.3f' % math.sqrt(mse))
    print('Avg Bias: %.3f' % bias)
    print('Avg Variance: %.3f' % var)
    
    return model


# #### ***Model 1: Linear regression***

# In[55]:


# Model for linear regression. Normalizing as first step with a single unit and layer to get a linear regression
linear_model = model_definition_compilation_fit(
    'linear model',
    layers=[
        normalizer,
        layers.Dense(units=1)
    ],
    learning_rate=0.1,
    epochs=5,
    validation_split=0.2
)


# #### *Comparing linear regression predictions with real values*

# In[69]:


test_predictions = linear_model.predict(test_features).flatten()

plt.clf()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [price]')
plt.ylabel('Predictions [price]')
plt.show()


# #### ***Model 2: Simple neural network***

# In[66]:


# Model for simple neural network. Normalizing as first step with two layers of intermediate computation and an output layer of a single cell
nn_model = model_definition_compilation_fit(
    'neural_net',
    layers=[
        normalizer,
        layers.Dense(units=64, activation='relu'),
        layers.Dense(units=64, activation='relu'),
        layers.Dense(1)
    ],
    learning_rate=0.1,
    epochs=5,
    validation_split=0.2
)


# #### *Comparing simple neural network predictions with real values*

# In[75]:


test_predictions = dropout_large_nn_model.predict(test_features).flatten()

plt.clf()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [price]')
plt.ylabel('Predictions [price]')
plt.xlim((0,5e6))
plt.ylim((0,5e6))
plt.show()


# ---
# ## **Performance analysis**
# After analyzing the error and the bias and the variance, it can be concluded that a neural network predicts better the relation between the independent variable and the dependent variables; and this makes sense, since we expect the performance of a neural network to be better than a simple regressio model.

# --- 
# ## **Regularization of the model**
# Having the neural network to work as the model for the prediction, we can now apply regularization techniques or adjustment of parameters to improve the performance of the neural network.
# <br>There were made 3 changes in the hyperparameters:
# - Large neural network (increasing the number of layers)
# - Simple neural network with dropout layer (to reduce overfitting)
# - Large neutal network with droput layers

# #### ***Model 2a: Large neural network***
# Increasing the number of layers

# In[67]:


# Model for big neural network. Normalizing as first step with four layers of intermediate computation and an output layer of a single cel
large_nn_model = model_definition_compilation_fit(
    'large_neural_net',
    layers=[
        normalizer,
        layers.Dense(units=40, activation='relu'),
        layers.Dense(units=40, activation='relu'),
        layers.Dense(units=40, activation='relu'),
        layers.Dense(units=40, activation='relu'),
        layers.Dense(units=1)
    ],
    learning_rate=0.1,
    epochs=5,
    validation_split=0.2
)


# #### ***Model 2b: Simple neural network with dropout layer***
# Adding a droput layer to avoid overfitting

# In[60]:


# Model for simple neural network. Normalizing as first step with two layers of intermediate computation, 1 dropout layer (to reduce overfitting) and an output layer of a single cell
dropout_nn_model = model_definition_compilation_fit(
    'dropout_neural_net',
    layers=[
        normalizer,
        layers.Dropout(.1),
        layers.Dense(units=70, activation='relu'),
        layers.Dense(units=70, activation='relu'),
        layers.Dense(1)
    ],
    learning_rate=0.1,
    epochs=5,
    validation_split=0.2
)


# #### ***Model 2c: Large neural network with dropout layers***
# Increasing the number of layers and adding dropout layers

# In[61]:


# Model for big neural network. Normalizing as first step with four layers of intermediate computation, 3 dropout layers (to reduce overfitting due to the big size of the network) and an output layer of a single cel
dropout_large_nn_model = model_definition_compilation_fit(
    'dropout_large_neural_net',
    layers=[
        normalizer,
        layers.Dropout(.1),
        layers.Dense(units=40, activation='relu'),
        layers.Dropout(.02),
        layers.Dense(units=40, activation='relu'),
        layers.Dropout(.02),
        layers.Dense(units=40, activation='relu'),
        layers.Dense(units=40, activation='relu'),
        layers.Dense(units=1)
    ],
    learning_rate=0.1,
    epochs=5,
    validation_split=0.2
)


# ---
# ## **Validation of results**
# Comparing all the models, we can conclude that the model that fits the best is the dropout large neural network since the mean absolute error, the bias and the variance are the lowest.

# In[68]:


# Comparing MSE for the different models
display(pd.DataFrame(test_results, index=['Mean absolute error [price]']).T)


# In[75]:


test_predictions = dropout_large_nn_model.predict(test_features).flatten()

plt.clf()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [price]')
plt.ylabel('Predictions [price]')
plt.xlim((0,5e6))
plt.ylim((0,5e6))
plt.show()