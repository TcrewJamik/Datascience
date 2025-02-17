import streamlit as st
import pandas as pd
import plotly.express as px


st.title('🎈 App Name')

st.write('Hello world!')
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')

with st.expander('Data'):
  st.write('X')
  X_raw = df.drop('species', axis = 1)
  st.dataframe(X_raw)

  st.write('y')
  y_raw = df.species
  st.dataframe(y_raw)

with st.sidebar:
  st.header('Введите признаки:')
  island = st.selectbox('Island', ('Torgersen', 'Dream', 'Biscoe'))
  bill_length_mm = st.slider('Bill length(mm)', 32.1, 59.6, 44.5)
  bill_depth_mm = st.slider('Bill depth(mm)', 13.1, 21.5, 17.3)
  flipper_length_mm = st.slider('Flipper length(mm)', 32.1, 59.6, 44.5)
  body_mass_g = st.slider('Body mass(g)', 32.1, 59.6, 44.5)
  gender = st.selectbox('Gender', ('female', 'male'))
  

st.subheader('Data Visualization')
fig = px.scatter(
  df, 
  x = 'bill_length_mm', 
  y = 'bill_depth_mm',
  color = 'island', 
  title = 'Bill Length vs Bill Depth by Island')

st.plotly_chart(fig)

fig2 = px.histogram(
  df, 
  x = 'body_mass_g',
  nbins = 30,
  title = 'Distribution')

st.plotly_chart(fig2)

input_df = pd.DataFrame(df, index=[0])
input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
  st.write('**Imput penguin**')
  st.dataframe(input_df)
  st.write('**Combined penguins data** (Input row + original data)')
  st.dataframe(input_penguins)

encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix = encode)

X = df_penguins[1:]
input_row = df_penguins[1:]

target_mapper = {'Adelit': 0, 'Chinstrap': 1, 'Gentoo': 2}
def target_encode(val):
  return target_mapper[val]
y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input penguin)**')
  st.dataframe(input_row)
  st.write('**Encoded y**')
  st.write(y)
