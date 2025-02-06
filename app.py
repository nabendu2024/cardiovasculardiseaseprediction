
import streamlit as st
import numpy as np
import pickle

# Load model and transformer
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('transformer.pkl', 'rb') as file:
    pt = pickle.load(file)

def prediction(input_list):
    input_list = np.array(input_list, dtype=object)
    pred = model.predict_proba([input_list])[:, 1][0]
    result = ("You Have More Chances of Getting Diseased" if pred > 0.5 
              else "You Have Less Chances of Getting Diseased")
    return f"{result}\nYour Probability Of Having Cardio Vascular Disease is {round(pred,2)}\nTake Care"

def main():
    st.title('CardioLens: Analyzing Lifestyle Data to Predict Heart Disease Risks')
    
    # Numeric inputs
    h = st.number_input('What is Your Height (in cm)?', min_value=50.0, max_value=250.0, step=0.1)
    w = st.number_input('What is Your Weight (in Kg)?', min_value=20.0, max_value=200.0, step=0.1)
    bmi = st.number_input('Tell us about Your Body Mass Index?', min_value=10.0, max_value=50.0, step=0.1)
    
    # Categorical inputs
    gh = st.selectbox('What About Your General Health?', ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    gh_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}
    gh = gh_map[gh]
    
    bg = st.selectbox('What is Your Blood Group?', ['A', 'B', 'O', 'AB'])
    bg_map = {'O': 0, 'B': 1, 'A': 2, 'AB': 3}
    bg = bg_map[bg]
    
    age = st.slider('Tell us How old are you?', min_value=18, max_value=100, step=1)
    age = 0 if age < 30 else 1 if age < 50 else 2
    
    ckup = st.selectbox("When was the last time you had a check-up?", 
                        ['Never', 'Within the past year', 'Within the past 2 years', 'Within the past 5 years'])
    ckup_map = {'Never': 0, 'Within the past year': 1, 'Within the past 2 years': 2, 'Within the past 5 years': 3}
    ckup = ckup_map[ckup]
    
    smk = 1 if st.selectbox('Do You Smoke?', ['Yes', 'No']) == 'Yes' else 0
    sex = 1 if st.selectbox('Tell Us About Your Gender?', ['Male', 'Female']) == 'Male' else 0
    dp = 0 if st.selectbox('Tell Us About Your Diet Preference?', ['Veg', 'Non Veg']) == 'Veg' else 1
    
    gv = st.slider('Rate Your Green Vegetable Consumption (Low: 1, High: 4)', 1, 4, step=1)
    fr = st.slider('Rate Your Fruit Consumption (Low: 1, High: 4)', 1, 4, step=1)
    fry = st.slider('Rate Your Fried Food Consumption (Low: 1, High: 4)', 1, 4, step=1)
    alco = st.slider('Rate Your Alcohol Consumption (Low: 1, High: 4)', 1, 4, step=1)
    
    dep = 1 if st.selectbox('Are You Feeling Depressed?', ['Yes', 'No']) == 'Yes' else 0
    mar = 1 if st.selectbox('What About Your Marital Status?', ['Married', 'UnMarried']) == 'UnMarried' else 0
    exer = 1 if st.selectbox('Do you work out every day?', ['Yes', 'No']) == 'Yes' else 0
    
    skin = 1 if st.selectbox('Have You Ever Been Diagnosed With Skin Cancer?', ['Yes', 'No']) == 'Yes' else 0
    other = 1 if st.selectbox('Have You Ever Been Diagnosed With Any Type Of Cancer?', ['Yes', 'No']) == 'Yes' else 0
    arth = 1 if st.selectbox('Are You Suffering from Arthritis?', ['Yes', 'No']) == 'Yes' else 0
    diab = 1 if st.selectbox('Do You Have Any Type Of Diabetes?', ['Yes', 'No']) == 'Yes' else 0
    vac = 1 if st.selectbox('Are You Vaccinated Against Cardiovascular Disease?', ['Yes', 'No']) == 'Yes' else 0
    
    # Transform numerical data
    tran_data = pt.transform([[h, w, bmi]])
    h_t, w_t, bmi_t = tran_data[0]
    
    input_list = [h_t, w_t, gh, bg, age, bmi_t, ckup, smk, sex, dp, gv, fr, fry, alco, dep, mar, exer, skin, other, arth, diab, vac]
    
    if st.button('Show Prediction'):
        response = prediction(input_list)
        st.success(response)

if __name__ == '__main__':
    main()
                                        
    
    

