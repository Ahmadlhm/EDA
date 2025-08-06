import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.markdown('''
# **The EDA App**

This is the **EDA App** created in Streamlit.
It performs Exploratory Data Analysis (EDA) on a given dataset.
You can upload your CSV file and visualize the data using various plots.
''')

# Upload CSV data
with st.sidebar.header('Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Dataframe shape:", df.shape)
    st.write("Dataframe columns:", df.columns)

    # Display the first few rows of the dataframe
    if st.checkbox('Show raw data'):
        st.subheader('Raw Data')
        st.write(df.head())

    # Display basic statistics
    if st.checkbox('Show basic statistics'):
        st.subheader('Basic Statistics')
        st.write(df.describe())

    # Display correlation matrix
    if st.checkbox('Show correlation matrix'):
        st.subheader('Correlation Matrix')
        numerical_features = df.select_dtypes(include=['int64', 'float64'])
        corr_matrix = numerical_features.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
    # Display distribution of numerical features
    if st.checkbox('Show distribution of numerical features'):
        st.subheader('Distribution of Numerical Features')
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        discrete_var = [var for var in num_cols if df[var].nunique() < 20]
        for col in discrete_var:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df[col])
            plt.title(f'Distribution of {col}')
            st.pyplot(fig)    

    # Display pairplot
    if st.checkbox('Show pairplot'):
        st.subheader('Pairplot')
        target_col = st.selectbox('Select target column for pairplot', df.columns)
        if target_col:
            g = sns.pairplot(df, hue=target_col)
            st.pyplot(g.fig)
    # Display count of categorical features        
    if st.checkbox('Show count of categorical features'):
        st.subheader('Count of Categorical Features')
        cat_cols = df.select_dtypes(include=[object]).columns.tolist()
        for col in cat_cols:
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(y=df[col])
            plt.title(f'Count of {col}')
            st.pyplot(fig)
    # Display missing values
    if st.checkbox('Show missing values'):
        st.subheader('Missing Values')
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])
        fig = plt.figure(figsize=(10, 4))
        sns.heatmap(df.isnull(), cbar=True, cmap='binary')
        plt.title('Missing Values Heatmap')
        st.pyplot(fig)
    # Display unique values in each column
    if st.checkbox('Show unique values in each column'):
        st.subheader('Unique Values in Each Column')
        unique_values = {col: df[col].nunique() for col in df.columns}
        st.write(unique_values)
    # Display boxplot for outliers
    if st.checkbox('Show boxplot for outliers'):
        st.subheader('Boxplot for Outliers')
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            fig = plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
            st.pyplot(fig)        

    # Display bar plot for categorical features
    if st.checkbox('Show bar plot for categorical features'):
        st.subheader('Bar Plot for Categorical Features')
        cat_cols = df.select_dtypes(include=[object]).columns.tolist()
        if cat_cols:
            col = st.selectbox('Select categorical feature', cat_cols)
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(y=df[col])
            plt.title(f'Bar Plot of {col}')
            st.pyplot(fig)
        else:
            st.write("No categorical features available for bar plot.") 

    # Display distribution of target with feature
    if st.checkbox('Show distribution of target with feature'):
        st.subheader('Distribution of Target with Feature')
        target_col = st.selectbox('Select target column', df.columns)
        feature_col = st.selectbox('Select feature column', df.columns)
        if target_col and feature_col:
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(x=target_col, hue=feature_col, data=df)
            plt.title(f'Distribution of {target_col} with {feature_col}')
            st.pyplot(fig)
        else:
            st.write("Please select both target and feature columns.")   

    # Display categories of selected numeric features using qcut
    if st.checkbox('Show categories of selected numeric features using qcut'):
        st.subheader('Categories of Numeric Features using qcut')
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        discrete_var = [var for var in num_cols if df[var].nunique() > 20]
        if discrete_var:
            col = st.selectbox('Select numeric feature', discrete_var)
            target_col = st.selectbox('Select target column', df.columns)
            if col and target_col:
                df[col + '_category'] = pd.qcut(df[col], q=4, labels=False)
                fig = plt.figure(figsize=(10, 4))
                sns.barplot(x=df[col + '_category'], y=target_col, data=df)
                plt.title(f'Categories of {col} ')
                st.pyplot(fig)
        else:
            st.write("No numeric features available for categorization.")

    # Display Counts of discrete Variables Grouped by Target
    if st.checkbox('Show counts of discrete variables grouped by target'):
        st.subheader('Counts of Discrete Variables Grouped by Target')
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist() 
        discrete_vars = [var for var in numerical_features if df[var].nunique() < 20]
        target_col = st.selectbox('Select target column', df.columns)
        if discrete_vars and target_col:
            for var in discrete_vars:
                ct = pd.crosstab(df[var], df[target_col])
                fig = plt.figure(figsize=(10, 4))
                ct.plot(kind='bar')
                plt.xlabel(var)
                plt.ylabel('Count')
                plt.title(f'Counts of {var} Variable Grouped by Target')
                st.pyplot(fig)
        else:
            st.write("No discrete variables or target column selected.")                                                              
else:
    st.info('Awaiting for CSV file to be uploaded.')


