import streamlit as st
import pandas as pd

# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import preprocessing as pp
from sklearn import utils

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


@st.cache
def getdata(filename, engine):
    taxi_data = pd.read_parquet(filename, engine=engine)
    return taxi_data


with header:
    st.title("Welcome to the data science project ")
    st.text('In this project I look into the transaction of taxis in NY city ...')

with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset on https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page')

    # taxi_data = pd.read_parquet('data/tripdata.parquet', engine='pyarrow')

    taxi_data_raw = getdata('data/tripdata.parquet', 'pyarrow')
    taxi_data = taxi_data_raw.head(3000)
    st.write(taxi_data.head(20))

    st.subheader('Pick-up location ID distribution on the NYC dataset')
    pulocation_dist = pd.DataFrame(
        taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)

with features:
    st.header('The features I created')

    st.markdown('* **first feature:** I created this feature due to ...')
    st.markdown('* **second feature:** I created this feature due to ...')

with model_training:
    st.header('Time to train the model')
    st.text('Here you get to choose the hyper-parameters of the model and see how the performance changes')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What could be the max_depth of the model?',
                               min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should be there?', options=[
                                     100, 200, 300, 'No Limit'], index=0)

    sel_col.text('Here is a list of features in my data:')
    sel_col.write(taxi_data.columns)

    input_feature = sel_col.text_input(
        'Which feature should be used as the input feature?', 'PULocationID')

   # regr = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    if n_estimators == 'No Limit':
        RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(
            max_depth=max_depth, n_estimators=n_estimators)

    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]

    # st.write(X)
    # st.write(y)

    X = scale(X)
    y = scale(y)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.10)

    regr.fit(Xtrain, ytrain)
    # prediction = regr.predict(y)
    prediction = regr.predict(Xtest)

    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(ytest, prediction))

    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_squared_error(ytest, prediction))

    disp_col.subheader('R squared score of the model is:')
    disp_col.write(r2_score(ytest, prediction))
