import pandas as pd
import numpy as np
import streamlit as st
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from streamlit_option_menu import option_menu
st.set_page_config(page_title="ML", page_icon=":rocket:", layout="wide", initial_sidebar_state="expanded")

path = "C:/Users/Senthil/Desktop/DS/Projects Photo/Copper_image.jpg"
image1 = Image.open(path)

st.markdown("<h1 style='text-align: center; color: blue;'>Industrial Copper Modelling</h1>", unsafe_allow_html=True)

selected = option_menu(None, ["Home", "Classification", "Regression"],
                       icons=["house"],
                       orientation="horizontal")


with open("Model_C.pkl","rb") as file:
    Classification = pickle.load(file) 

P = pd.read_csv("C:/Users/Senthil/Desktop/DS/Code/Copper_ML/F_C_Copper.csv")  
X = P[["quantity tons","customer","product_ref"]]
Y = P[["status"]] 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
K = Classification.predict(X_test)
Accuracy = accuracy_score(y_test, K)
f1 = f1_score(y_test, K)

with open("Model_R.pkl","rb") as file:
    Regression = pickle.load(file) 

p = pd.read_csv("C:/Users/Senthil/Desktop/DS/Code/Copper_ML/F_R_Copper.csv") 
X = p[["quantity tons","thickness","width","item type","product_ref","year","day","month","application","customer",'country']]
Y = p[["selling_price"]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
K = Regression.predict(X_test)
f2 = r2_score(y_test, K)

if selected == 'Home':
    col4,col5 = st.columns(2)
    col4.subheader(":orange[About]")
    col4.write(":green[ The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data. Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer . You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values. The solution must include the following steps: Exploring skewness and outliers in the dataset. Transform the data into a suitable format and perform any necessary cleaning and pre-processing steps. ML Regression model which predicts continuous variable ‘Selling_Price’. ML Classification model which predicts Status: WON or LOST. Creating a streamlit page where you can insert each column value and you will get the Selling_Price predicted value or Status(Won/Lost).]")
    col5.markdown("""         """)
    col5.markdown("""         """)
    col5.markdown("""         """)
    col5.markdown("""         """)
    col5.image(image1,width = 700)    
    
   
if selected == "Classification":
  st.write(":red[RandomForestClassification]")
  if st.button("Click To Check Accuracy"):
    st.markdown(f'<h1 style="text-align: center; color:blue;">Accuracy Score: {Accuracy:.5%}</h1>', unsafe_allow_html=True)
    
    
if selected == "Classification":
  col1,col2 = st.columns(2)
  A = col1.number_input("Enter Tons", min_value = 0,value=None)
  D = col2.selectbox("Tons(EXAMPLE)",[54.15113862 ,768.0248392 , 386.12794891, 202.41106541, 785.52626157,225.79067611 ,113.38712363 ,69.0718528 , 630.6269167 , 113.99566554,27.51254472 , 32.23531688 , 35.31033249 , 75.12439366 , 20.80711354,53.58103351  ,71.36697792, 471.83818036 , 53.72974562 , 79.33399031,179.58279605 ,927.43074747 , 99.05919889 ,185.14965601 ,102.42177284,75.75766317, 179.7245138 , 207.77670651 , 29.01093671, 92.7172555,44.77285681 , 52.65787989  ,59.90735531, 540.75491108,  56.8313529,49.7045555 , 181.91084181 , 26.53371627 ,132.88411415 ,126.77620144,27.74322116,  30.89994963, 371.65571216,  62.57489188  ,37.87714559,204.2589525 ,43.48741555 ,425.02618288, 901.30462426, 835.3378492 ])
  B = st.number_input("Enter Customer ID", min_value=0, max_value=1124, step=1,value=None)
  C = st.number_input("Enter Product reference ID", min_value=0, max_value=32, step=1,value=None)
  try:
    if A >=0 and B >=0 and C >=0: 
      with st.spinner("Please Wait..."):
        if st.button('Click Here To know Wheater Transaction WON or LOST'):
          Ans = Classification.predict([[A,B,C]])
          if Ans == 1:
              st.markdown('<h1 style="text-align: center; color:blue;">LOST</h1>', unsafe_allow_html=True)
          else:
              st.markdown('<h1 style="text-align: center; color:red;">LOST</h1>', unsafe_allow_html=True)
  except:
    st.warning("Please Enter Valid Value")
    
    
if selected == "Regression":
  st.write(":red[ExtraTreesRegressor]")
  if st.button("Click To Check Accuracy"):
    st.markdown(f'<h1 style="text-align: center; color:blue;">Accuracy Score: {f2:.5%}</h1>', unsafe_allow_html=True)
    
if selected == "Regression":
  col1,col2 = st.columns(2)
  A = col1.number_input("Enter Tons",min_value = 0.1,value=None)
  B = col2.number_input("Enter Thickness",min_value = 0.1,value=None)
  C = col1.number_input("Enter Width",min_value = 0,value=None)
  D = col2.number_input("Enter Item_Type", min_value=0, max_value=6, step=1,value=None)
  E = col1.number_input("Enter Product reference ID", min_value=0, max_value=32, step=1,value=None)
  F = col2.selectbox("Enter Year",["Select",2020,2021])
  G = col1.selectbox("Enter Month",["Select", 4,  3,  2,  1, 12, 11, 10,  9,  8,  7])
  H = col2.number_input("Enter day", min_value=0, max_value=31, step=1,value=None)
  I = col1.number_input("Enter Country", min_value=0, max_value=16, step=1,value=None)
  J = col2.number_input("Enter Customer ID", min_value=0, max_value=1124, step=1,value=None)
  K = col1.number_input("Enter Application ", min_value=0, max_value=29, step=1,value=None)
  try:
    if A is not None and B is not None and C >=0 and D >=0 and E >=0 and H >=0 and I >=0 and J >=0 and K >=0 and F is not 'Select' and G is not 'Select':
      Z = np.log(A)
      Y = np.log(B)
      with st.spinner("Please Wait..."):
        if st.button("Click To Predict"):
            U = Regression.predict([[Z,Y,C,D,E,F,H,G,K,J,I]])
            V = np.exp(U)
            for i, v in enumerate(V):
              st.markdown(f'<h1 style="text-align: center; color:blue;">Predicted Amount: {v:.5f}</h1>', unsafe_allow_html=True)
  except:
    st.warning("Please Enter Valid Value")
