import streamlit as st
import os
import random 

# importing libraries
import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt 
import seaborn as sn                   # For plotting graphs

import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
from numpy import sqrt

is_streamlit_cloud = os.environ.get("STREAMLIT_SERVER") is not None

if is_streamlit_cloud:
    # File path for deployment on Streamlit Cloud
    csv_file_path = "insurance.csv"
else:
    # File path for local development
    data = pd.read_csv(r"insurance.csv")

data = pd.read_csv(r"insurance.csv")
dummy = pd.get_dummies(data['smoker'])
df = pd.concat((data,dummy),axis = 1)
df = df.drop(['no'],axis = 1)
df.rename(columns = {'yes':'smokers_norm'}, inplace = True)
X = df[['age','smokers_norm']]
y = df['charges']
cols_not_reg3=['age', 'smokers_norm']
kf=KFold(n_splits=10, random_state=1, shuffle=True)
intercepts=[]
mses=[]
coefs=[]

for train_index, test_index in kf.split(df[cols_not_reg3]):
    
    lr=LinearRegression()
    lr.fit(df[cols_not_reg3].iloc[train_index],df["charges"].iloc[train_index])
    lr_predictions=lr.predict(df[cols_not_reg3].iloc[test_index])
    
    lr_mse=sklearn.metrics.mean_squared_error(df["charges"].iloc[test_index],lr_predictions)
    
    intercepts.append(lr.intercept_)
    
    coefs.append(lr.coef_)
    mses.append(lr_mse)



# 1. as sidebar menu
nav = st.sidebar.radio("Main Menu",["Home","Prediction","Contribute"])
if nav == "Home":
    st.header("HEALTH INSURANCE PREDICTION")
    st.image("https://cdn.pixabay.com/photo/2022/06/15/09/50/health-insurance-7263536_960_720.png")
    st.write("Below is the dataframe. In the dataframe, charges represent the cost of health insurance.")
    
    if st.checkbox("Show Table"):
        st.write(data)
        st.text("In the dataframe, we have variables like 'age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'")
    
        st.subheader("Correlation Visualizations")
        st.write("Select variables to visualize and compare their correlation with charges:")
    
        # Create a multiselect widget to select multiple variables
        variables_to_visualize = st.multiselect("Select variables", ["age", "bmi", "smoker", "children"])

        
    
        # Plot the correlation visualization based on the selected variables
        if variables_to_visualize:
            fig, ax = plt.subplots(figsize=(8, 6))
            for variable in variables_to_visualize:
                sns.scatterplot(x=variable, y="charges", data=data, ax=ax)
            plt.xlabel("Selected Variables")
            plt.ylabel("Charges")
            plt.legend()
            st.pyplot(fig)
        else:
            st.write("Select at least one variable to visualize.")


if nav == "Prediction":
    st.header("Know your Health Insurance cost")
    rmses = [x**.5 for x in mses]
    avg_rmse = np.mean(rmses)
    avg_intercept = np.mean(intercepts)
    age_coefs = []
    smoking_coefs = []

    for vals in coefs:
        age_coefs.append(vals[0])
        smoking_coefs.append(vals[1])

    age_coef = np.mean(age_coefs)
    smoking_coef = np.mean(smoking_coefs)
    avg_intercept = np.mean(intercepts)

    st.write("Enter the following details:")
    
    age_input = st.number_input("What is your age", 1, 80)
    smoking_input = st.selectbox('Are you a smoker or non-smoker?', ('Smoker', 'Non-smoker'))

    selected_smoking = 1 if smoking_input == 'Smoker' else 0

    pred = (age_coef * age_input) + (smoking_coef * selected_smoking) + avg_intercept

    if st.button("Predict"):
        st.write('You have selected:', 'Smoker' if selected_smoking == 1 else 'Non-smoker')
        st.success(f"Your predicted Health Insurance cost is {round(pred)} rupees")

if nav == "Contribute":
    st.header("Contribute to Health Insurance Prediction")
    st.write("We welcome contributions to improve this health insurance prediction app!")
    st.write("Here are a few ways you can contribute:")
    
    st.markdown("- Enhance the UI/UX design to make the app more visually appealing.")
    st.markdown("- Add more features such as different regression models or additional input variables.")
    st.markdown("- Improve error handling and input validation for a smoother user experience.")
    st.markdown("- Optimize the code for better performance.")
    st.markdown("- Share your feedback and suggestions for further improvements.")
    
    st.write("To contribute, you can:")
    st.markdown("- Fork the GitHub repository: https://github.com/Rohit-Gaikwad16/")
    st.markdown("- Make your changes and improvements.")
    st.markdown("- Submit a pull request to the main repository.")
    
    st.write("Thank you for your interest in contributing!")
