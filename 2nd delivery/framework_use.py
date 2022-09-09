#!/usr/bin/env python
# coding: utf-8

# # Extract

# ### Imports

# In[2]:


# !pip install pandas numpy matplotlib dataprep sklearn
# !pip install tensorflow-gpu


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Function that creates a report of the DataFrame (overview, variables, quantile statistics, descriptive statistics, correlations, missing values)
from dataprep.eda import create_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, regularizers, models

np.set_printoptions(precision=3, suppress=True)


# In[4]:


print('Tensorflow: ', tf.__version__)


# ## Get data
# https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

# In[5]:


# Reading data via Pandas from CSV
house_data = pd.read_csv('dataset/kc_house_data.csv')


# In[6]:


house_data.info()


# In[7]:


# Creation of the DF report and saving to HTML
report = create_report(house_data, title = 'House data report')


# In[8]:


report.save('House report.html')


# # Transform

# ## Date transformation

# In[9]:


# Extracting month and year from date column
house_data.loc[:, 'month'] = pd.DatetimeIndex(house_data['date']).month
house_data.loc[:, 'year'] = pd.DatetimeIndex(house_data['date']).year


# ## Column selection

# In[10]:


# Dropping location-related columns (lat, long, zipcode) and date (since it was processed in the last step)
house_data = house_data.drop(['id', 'lat', 'long', 'zipcode', 'date'], axis=1, errors='ignore')
print(f"Original data shape: {house_data.shape}")


# In[11]:


house_data


# ## Split into train and test sets

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


# # Model

# In[54]:


# Defining array to store the predictions of each model
test_results = {}


# In[25]:


# Function for plotting the loss history for each model
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error [Price]')
  plt.legend()
  plt.grid(True)


# ## Normalization

# In[26]:


# Normalization fitting for train features (X)
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())


# ## Model 1: Linear regression

# In[27]:


# Model definition. Normalizing as first step with a single unit and layer to get a linear regression
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])


# In[28]:


# Model compilation using MSE
linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[29]:


get_ipython().run_cell_magic('time', '', '# Model fit using training features and labels\nhistory = linear_model.fit(\n    train_features,\n    train_labels,\n    epochs=100,\n    # Suppress logging.\n    verbose=1,\n    # Calculate validation results on 20% of the training data.\n    validation_split = 0.2)\n')


# In[30]:


# Plotting the loss history for the model
plt.clf()
plot_loss(history)
plt.show()


# In[31]:


# Adding the model predictions to the previously defined dictionary
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=1)


# ## Model 2: Simple neural network

# In[39]:


# Model definition. Normalizing as first step with two layers of intermediate computation and an output layer of a single cell
nn_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(1)
])
nn_model.summary()


# In[40]:


# Model compilation using MSE
nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[41]:


get_ipython().run_cell_magic('time', '', '# Model fit using training features and labels\nhistory = nn_model.fit(\n    train_features,\n    train_labels,\n    epochs=100,\n    # Suppress logging.\n    verbose=1,\n    # Calculate validation results on 20% of the training data.\n    validation_split = 0.2)\n')


# In[42]:


# Plotting the loss history for the model
plt.clf()
plot_loss(history)
plt.show()


# In[43]:


# Adding the model predictions to the previously defined dictionary
test_results['neural_net'] = nn_model.evaluate(
    test_features, test_labels, verbose=1)


# ## Model 3: Large neural network

# In[47]:


# Model definition. Normalizing as first step with four layers of intermediate computation, 3 dropout layers (in an attemp to prevent overfitting) and an output layer of a single cell
large_nn_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=32, activation='relu',
                 kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.L2(1e-4),
                activity_regularizer=regularizers.L2(1e-5)
                ),
    layers.Dropout(.05),
    layers.Dense(units=32, activation='relu',
                 kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.L2(1e-4),
                activity_regularizer=regularizers.L2(1e-5)
                ),
    layers.Dropout(.05),
    layers.Dense(units=32, activation='relu',
                 kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.L2(1e-4),
                activity_regularizer=regularizers.L2(1e-5)
                ),
    layers.Dropout(.05),
    layers.Dense(units=32, activation='relu',
                 kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.L2(1e-4),
                activity_regularizer=regularizers.L2(1e-5)
                ),
    layers.Dense(units=1)
])

large_nn_model.summary()


# In[48]:


# Model compilation using MSE
large_nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[49]:


get_ipython().run_cell_magic('time', '', '# Model fit using training features and labels\nhistory = large_nn_model.fit(\n    train_features,\n    train_labels,\n    epochs=100,\n    # Suppress logging.\n    verbose=1,\n    # Calculate validation results on 20% of the training data.\n    validation_split = 0.2)\n')


# In[50]:


# Plotting the loss history for the model
plt.clf()
plot_loss(history)
plt.show()


# In[51]:


# Adding the model predictions to the previously defined dictionary
test_results['large_neural_net'] = large_nn_model.evaluate(
    test_features, test_labels, verbose=1)


# ## Validation of results

# In[58]:


# Comparing MSE for the different models
display(pd.DataFrame(test_results, index=['Mean absolute error [price]']).T)


# ### Comparing linear regression predictions with real values

# In[45]:


test_predictions = linear_model.predict(test_features).flatten()

plt.clf()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [price]')
plt.ylabel('Predictions [price]')
plt.show()


# ### Comparing simple neural network predictions with real values

# In[46]:


test_predictions = nn_model.predict(test_features).flatten()

plt.clf()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [price]')
plt.ylabel('Predictions [price]')
plt.show()


# ### Comparing large neural network predictions with real values

# In[53]:


test_predictions = large_nn_model.predict(test_features).flatten()

plt.clf()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [price]')
plt.ylabel('Predictions [price]')
plt.show()


# In[ ]:




