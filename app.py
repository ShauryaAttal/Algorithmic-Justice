import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import plotly.express as px
from joblib import dump, load

@st.cache_data
# 2. Data Preprocessing data function goes here (including the df, X_train,...) -------------------------------------------
def load_data():
    # Load the dataset from a CSV file.
    data = pd.read_csv("compas-scores-two-years.csv")

    # Drop unnecessary columns that are not required for the analysis or model training.
    df = data.drop(labels=['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'days_b_screening_arrest',
                           'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date', 'c_arrest_date', 'c_days_from_compas',
                           'r_case_number', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc',
                           'r_jail_in', 'r_jail_out', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'decile_score.1',
                           'violent_recid', 'vr_charge_desc', 'in_custody', 'out_custody', 'priors_count.1', 'start', 'end',
                           'v_screening_date', 'event', 'type_of_assessment', 'v_type_of_assessment', 'screening_date',
                           'score_text', 'v_score_text', 'v_decile_score', 'decile_score', 'is_recid', 'is_violent_recid'], axis=1)

    # Rename columns for clarity and easier understanding.
    df.columns = ['sex', 'age', 'age_category', 'race', 'juvenile_felony_count', 'juvenile_misdemeanor_count', 'juvenile_other_count',
                  'prior_convictions', 'current_charge', 'charge_description', 'recidivated_last_two_years']

    # Filter out charge descriptions that are too rare to be statistically significant.
    value_counts = df['charge_description'].value_counts()
    df = df[df['charge_description'].isin(value_counts[value_counts >= 70].index)].reset_index(drop=True)

    # Convert categorical columns to one-hot encoded vectors for machine learning compatibility.
    for colname in df.select_dtypes(include='object').columns:
        one_hot = pd.get_dummies(df[colname])
        df = df.drop(colname, axis=1)
        df = df.join(one_hot)

    # Separate the dataset into features (X_all) and the target variable (y_all).
    y_column = 'recidivated_last_two_years'
    X_all, y_all = df.drop(y_column, axis=1), df[y_column]

    # Split the dataset into training and testing sets for model evaluation.
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3)

    # Return the processed dataframe and split datasets.
    return df, X_train, X_test, y_train, y_test

# Call the function to load the data and get the datasets for model training and evaluation.
df, X_train, X_test, y_train, y_test = load_data()

@st.cache_resource
# 3. Function to load the model --------------------------------------------------------------------------------------
def load_model():
    model = load('./THGBmodelamazing.joblib')
    return model

model = load_model()

# 4. Our Site Design
# Streamlit title page configuration-------------------------------------------------------------------------------
st.title("Criminal Justice Neural Network Model")
st.write("Explore the COMPAS data and predict recidivism.")

# Sidebar for user input-------------------------------------------------------------------------------------------
st.sidebar.title("User Input for Predictions")
# Assuming 'age' and 'prior_convictions' are two features you'd like the user to input for predictions
age = st.sidebar.number_input('Age', min_value=18, max_value=70, value=30)
prior_convictions = st.sidebar.number_input('Prior Convictions', min_value=0, max_value=100, value=0)

# Process user inputs for prediction-------------------------------------------------------------------------------
def process_input(age, prior_convictions):
    input_data = pd.DataFrame([[0] * len(X_train.columns)], columns=X_train.columns)
    input_data['age'] = age
    input_data['prior_convictions'] = prior_convictions

    return input_data

# Predictions-------------------------------------------------------------------------------------------------
if st.sidebar.button('Predict'):
    input_data = process_input(age, prior_convictions)
    prediction = model.predict(input_data)
    st.sidebar.write(f'Prediction: {"High Risk" if prediction[0] == 1 else "Low Risk"}')

# Aggregate one-hot encoded race columns
def aggregate_race_columns(df):
    races = ['African-American', 'Asian', 'Caucasian', 'Hispanic', 'Native American', 'Other']
    df_race_aggregated = df[races].idxmax(axis=1)
    return df_race_aggregated

# Interactive Data Visualizations
st.header("Data Visualizations")
viz_option = st.selectbox("Choose a Visualization", ["Race Distribution", "Age Distribution", "Prior Convictions Distribution"])

if viz_option == "Race Distribution":
    race_data = aggregate_race_columns(df)
    fig = px.histogram(race_data, x=race_data, color=race_data, title="Race Distribution in COMPAS Data")
    st.plotly_chart(fig)
elif viz_option == "Age Distribution":
    fig = px.histogram(df, x="age", title="Age Distribution in COMPAS Data")
    st.plotly_chart(fig)
elif viz_option == "Prior Convictions Distribution":
    fig = px.histogram(df, x="prior_convictions", title="Prior Convictions Distribution")
    st.plotly_chart(fig)

st.header("Model Analysis and Fairness Metrics")
if st.checkbox('Show Metrics and Fairness Analysis'):
    # Additional code to display model analysis and fairness metrics
    # Confusion Matrix, Classification Report, etc.
    # Example: Confusion Matrix for African-American group

    race_column = 'African-American'

    group = X_test[X_test[race_column] == 1]
    y_true_group = y_test[group.index]
    y_pred_group = model.predict(group)
    cm = confusion_matrix(y_true_group, y_pred_group)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="d")
    ax.set_title('Confusion Matrix for '+race_column+' Group')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)

    # Precision, Recall, F1 Score for the group
    st.write("Precision ("+race_column+"):", precision_score(y_true_group, y_pred_group, average="macro"))
    st.write("Recall ("+race_column+"):", recall_score(y_true_group, y_pred_group, average="macro"))
    st.write("F1 Score ("+race_column+"):", f1_score(y_true_group, y_pred_group, average="macro"))
