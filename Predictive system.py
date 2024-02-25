# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 23:54:49 2024

@author: rajti
"""

import numpy as np
import pickle
from sklearn.metrics import accuracy_score


load_model = pickle.load(open('C:/Users/rajti/Desktop/Excelr project/trained_model_new1.save', 'rb'))
input_data =( 1,0,0.5,0.5,1,1 )
input_data_as_numpy_array = np.asarray(input_data)
input_data_as_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = load_model.predict( input_data_as_reshaped )

print(prediction)

 
# Example data (replace with your actual data)
y_true = [1]  # Ground truth labels (actual class)
y_pred = prediction  # Predicted labels

# Calculate accuracy score
accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy Score: {accuracy:.2f}")

# 0--bankruptcy
# 1--non-bankruptcy

if ( prediction[0] == 0):
  print(" bankruptcy ")

else:
    print(" non-bankruptcy ")