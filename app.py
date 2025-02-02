import streamlit as st
import pandas as pd
import numpy as np

# Title of the app
st.title('Simple Streamlit App')

# Create a slider widget
x = st.slider('Select a value for x', 0, 100)

# Display the selected value
st.write(f'The selected value is {x}')

# Create a DataFrame
df = pd.DataFrame({
    'Column 1': np.random.randn(10),
    'Column 2': np.random.randn(10)
})

# Display the DataFrame
st.write('Here is a random DataFrame:')
st.write(df)