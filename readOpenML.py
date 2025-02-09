import openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import streamlit as st

# Title of the app
st.title('OpenML Dataset and ML Model')

# Fetch the Iris dataset from OpenML
st.write('Fetching the Iris dataset from OpenML...')
dataset = openml.datasets.get_dataset(61)

# Get the data as a pandas DataFrame
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Display the dataset
st.write('Here is the dataset:')
st.write(pd.concat([X, y], axis=1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy of the model: {accuracy:.2f}')