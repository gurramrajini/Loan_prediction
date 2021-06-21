import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
import streamlit as st
import pickle

st.title('Loan Prediction Model')

st.sidebar.header('User Input Parameters')


# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
def get_loan_prediction(Gender,Married,Dependents,Education,Self_Employed,LoanAmount,Loan_Amount_Term,ApplicantIncome,CoapplicantIncome,Credit_History,Property_Area):
    
    if Property_Area == 'Rural':
        Property_Area = 0
    elif Property_Area == 'Semiurban':
         Property_Area = 1
    else:
         Property_Area = 2


    X = pd.DataFrame({'Gender': Gender, 'Married': Married,
                      'Dependents':Dependents,'Education':Education,'Self_Employed': Self_Employed,
                      'ApplicantIncome': ApplicantIncome,'CoapplicantIncome': CoapplicantIncome,
                      'LoanAmount':LoanAmount,'Loan_Amount_Term': Loan_Amount_Term,
                      'Credit_History': Credit_History,'Property_Area':Property_Area},index =[1])
    
    def Transforming(df):
        df['Dependents'] = df['Dependents'].str.rstrip('+')
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(np.int)
        df['Married'] = df['Married'].map({'No': 0, 'Yes': 1}).astype(np.int)
        df['Education'] = df['Education'].map({'Not Graduate': 0, 'Graduate': 1}).astype(np.int)
        df['Self_Employed'] = df['Self_Employed'].map({'No': 0, 'Yes': 1}).astype(np.int)
        df['Credit_History'] = df['Credit_History'].map({'No':0,'Yes':1}).astype(np.int)
        df['Dependents'] = df['Dependents'].astype(np.int)
        return df
    X = Transforming(X)
    prediction = classifier.predict(X)[0]
    st.subheader('User Input Parameters')
    st.write(X)
    prediction_proba = classifier.predict_proba(X)
    st.write(prediction_proba)
    
    
    return prediction


def main():
    Gender = st.sidebar.radio("Gender",('Male','Female'))
    Credit_History = st.sidebar.radio("Credit History",('Yes','No'))
    Married = st.sidebar.radio("Married",('No','Yes'))
    Dependents = st.sidebar.radio("Dependents",('0','1','2','3'))
    ApplicantIncome = st.sidebar.slider("Current Applicants Income",0,100000)
    CoapplicantIncome = st.sidebar.slider("Co-Applicants Income(if any)",0,50000)
    LoanAmount = st.sidebar.slider("Loan Amount( in thousand)", 0,50000)
    Loan_Amount_Term = st.sidebar.select_slider("Loan Period (in number of Months)",options = [12,36,60,84,120,180,240,300,360,480])
    
    Education = st.sidebar.radio("Education Level",('Graduate','Not Graduate'))
    Self_Employed = st.sidebar.radio("Self Employed",('Yes','No'))
    
    Property_Area = st.sidebar.radio("Property Area",("Rural",'Semiurban','Urban'))
    result = ""
    if st.button("Predict"):
        result = get_loan_prediction(Gender,Married,Dependents,Education,Self_Employed,LoanAmount,Loan_Amount_Term,ApplicantIncome,CoapplicantIncome,Credit_History,Property_Area)

    if result == 1:
        
            st.success('Loan Approved')
            
    elif result == 0:
        
        st.success('Loan Rejected')
    else:
        st.success('Decision Pending')
    

    

if __name__ == '__main__':
    main()
