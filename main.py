from sklearn import preprocessing
import streamlit as st
import pandas as pd
import os
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

path = os.path.dirname(__file__)
filename = path+'/final_model.save'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv(path+"/Clustered_Customer_Data.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("Mall Customer Segmentation Prediction")

with st.form("my_form"):
    gendre=st.number_input(label='Gendre',step=1)
    age=st.number_input(label='Age',step=1)
    annual_Income =st.number_input(label='Annual_Income',step=1)
    spending_Score =st.number_input(label='Spending_Score',step=1)


    data=[[gendre,age,annual_Income,spending_Score]]

    submitted = st.form_submit_button("Submit")

if submitted:
    clust=loaded_model.predict(data)[0]
    print('Data Belongs to Cluster',clust)

    cluster_df1=df[df['Cluster']==clust]
    plt.rcParams["figure.figsize"] = (20,3)
    for c in cluster_df1.drop(['Cluster'],axis=1):
        fig, ax = plt.subplots()
        grid= sns.FacetGrid(cluster_df1, col='Cluster')
        grid= grid.map(plt.hist, c)
        plt.show()
        st.pyplot(figsize=(5, 5))


