import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier  # Added import
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import plotly.express as px
from joblib import dump, load

# 2. Data Preprocessing function
@st.cache_data
def load_data():
    # Load the dataset from a CSV file.
    data = pd.read_csv("compas-scores-two-years.csv")

    # Drop unnecessary columns
    df = data.drop(labels=['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'days_b_screening_arrest',
                           'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date', 'c_arrest_date', 'c_days_from_compas',
                           'r_case_number', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc',
                           'r_jail_in', 'r_jail_out', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'decile_score.1',
                           'violent_recid', 'vr_charge_desc', 'in_custody', 'out_custody', 'priors_count.1', 'start', 'end',
                           'v_screening_date', 'event', 'type_of_assessment', 'v_type_of_assessment', 'screening_date',
                           'score_text', 'v_score_text', 'v_decile_score', 'decile_score', 'is_recid', 'is_violent_recid'], axis=1)

    # Rename columns for clarity
    df.columns = ['sex', 'age', 'age_category', 'race', 'juvenile_felony_count', 'juvenile_misdemeanor_count', 'juvenile_other_count',
                  'prior_convictions', 'current_charge', 'charge_description', 'recidivated_last_two_years']

    # Filter out rare charge descriptions
    value_counts = df['charge_description'].value_counts()
    df = df[df['charge_description'].isin(value_counts[value_counts >= 70].index)].reset_index(drop=True)

    # Convert categorical columns to one-hot encoded vectors
    for colname in df.select_dtypes(include='object').columns:
        one_hot = pd.get_dummies(df[colname])
        df = df.drop(colname, axis=1)
        df = df.join(one_hot)

    # Separate features and target
    y_column = 'recidivated_last_two_years'
    X_all, y_all = df.drop(y_column, axis=1), df[y_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3)

    return df, X_train, X_test, y_train, y_test

# Load the data
df, X_train, X_test, y_train, y_test = load_data()

# 3. Function to load or train the model
@st.cache_resource
def load_model(tuned=False, X_train=None, y_train=None):
    if tuned:
        # Hyperparameter tuning for the model
        modelHGBC = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=5,
            l2_regularization=0.1,
            random_state=1
        )
    else:
        # Base model
        modelHGBC = HistGradientBoostingClassifier(random_state=1)

    # Fit the model
    modelHGBC.fit(X_train, y_train)

    return modelHGBC

# Load or train the model
model = load_model(tuned=True, X_train=X_train, y_train=y_train)

# Evaluate the model
y_pred_hgbc = model.predict(X_test)
accuracy_hgbc = accuracy_score(y_test, y_pred_hgbc)

# Display accuracy in the Streamlit app
st.write(f"Tuned Histogram Gradient Boosting Classifier Model Accuracy: {round(accuracy_hgbc, 6) * 100}%")

# 4. Streamlit site design
st.title("Criminal Justice Neural Network Model")
st.write("Explore the COMPAS data and predict recidivism.")

# Sidebar for user input
st.sidebar.title("User Input for Predictions")
age = st.sidebar.number_input('Age', min_value=18, max_value=70, value=30)
prior_convictions = st.sidebar.number_input('Prior Convictions', min_value=0, max_value=100, value=0)

# Process user input for prediction
def process_input(age, prior_convictions):
    input_data = pd.DataFrame([[0] * len(X_train.columns)], columns=X_train.columns)
    input_data['age'] = age
    input_data['prior_convictions'] = prior_convictions

    # Ensure one-hot encoding is handled correctly for categorical columns
    input_data = pd.get_dummies(input_data, drop_first=True)
    return input_data

# Predictions
if st.sidebar.button('Predict'):
    input_data = process_input(age, prior_convictions)
    prediction = model.predict(input_data)
    st.sidebar.write(f'Prediction: {"High Risk" if prediction[0] == 1 else "Low Risk"}')

# Aggregate one-hot encoded race columns
def aggregate_race_columns(df):
    races = ['African-American', 'Asian', 'Caucasian', 'Hispanic', 'Native American', 'Other']
    df_race_aggregated = df[races].idxmax(axis=1)
    return df_race_aggregated

# Data visualizations
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

# Model analysis and fairness metrics
st.header("Model Analysis and Fairness Metrics")
if st.checkbox('Show Metrics and Fairness Analysis'):
    # Example: Confusion Matrix for African-American group
    race_column = 'African-American'

    group = X_test[X_test[race_column] == 1]
    y_true_group = y_test[group.index]
    y_pred_group = model.predict(group)
    cm = confusion_matrix(y_true_group, y_pred_group)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="d")
    ax.set_title(f'Confusion Matrix for {race_column} Group')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)

    # Precision, Recall, F1 Score for the group
    st.write(f"Precision ({race_column}):", precision_score(y_true_group, y_pred_group, average="macro"))
    st.write(f"Recall ({race_column}):", recall_score(y_true_group, y_pred_group, average="macro"))
    st.write(f"F1 Score ({race_column}):", f1_score(y_true_group, y_pred_group, average="macro"))
