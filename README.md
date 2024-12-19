# Mental Health Counseling Analysis

This repository contains the implementation of a project analyzing mental health counseling transcripts. The primary goals of this project are to profile and clean the dataset, perform exploratory data analysis (EDA), build predictive machine learning models, and create a web application for practical use. Below is an overview of the repository's contents and functionalities.

---

## Repository Structure

### 1. **Data Profiling**
- Folder: `data_profiling`
- **Description**: This folder contains scripts and notebooks for profiling and understanding the dataset. Python tools like `pandas` and `y_dataprofiling` have been used for detailed profiling to identify key data characteristics, outliers, and potential data quality issues.

### 2. **Data Processing and EDA**
- **Description**: Comprehensive data cleaning and exploratory analysis have been performed, including:
  - Handling missing values
  - Removing duplicates
  - Standardizing text and numeric fields
  - Visualizing trends and distributions
- **Tools Used**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `ydata_profiling`
- **Files**: The corresponding notebooks for these steps are available in the Airflow and Machine Learning Model folders.

### 3. **Machine Learning Models**
- Folder: `machine_learning_models`
- **Description**: Two machine learning models were built to infer useful insights from the dataset:
  1. **Classification Model 1**: Predicts the mental disorder of a person based on counseling transcripts using advanced natural language processing (NLP) techniques and machine learning algorithms.
     - Classes:
       - Bipolar Type-2
       - Depression
       - Normal
       - Bipolar Type-1
     - **Workflow**:
       - Preprocessing: Tokenization, text cleaning, and feature extraction (TF-IDF or embeddings).
       - Modeling: Trained using algorithms such as Logistic Regression, Random Forest, or Neural Networks.
       - Evaluation: Performance metrics include accuracy, precision, recall, and F1-score.
      
         
  2. **Classification Model 2**: Predicts the mood swings of a person using supervised machine learning techniques.
     - Classes:
       - High
       - Medium
       - Low
     - **Workflow**:
       - Preprocessing: Includes handling missing values, encoding categorical features, and feature scaling.
       - Modeling: Built using algorithms like Decision Trees, Gradient Boosting, or Support Vector Machines.
       - Evaluation: Performance metrics include confusion matrix, accuracy, and ROC-AUC scores.
- **Files**: The implementation details and training notebooks for these models are included in this folder.

### 4. **LLM-based application**


### 5. Web Application - Mental Health Counseling and LLM Integration

This web application is part of a larger project and focuses on providing personalized mental health assistance by leveraging a pre-trained large language model (LLM) and advanced similarity search techniques. The app combines user-friendly interfaces with powerful backend technologies to deliver relevant information and actionable insights for mental health challenges.


- **Interactive User Interface**: Built with **Streamlit**, enabling users to:
  - Enter free-text descriptions of mental health challenges and receive tailored suggestions via an LLM.
  - Search a database of mental health counseling data to retrieve relevant examples.
  - Assess their current mental health state and receive advice on managing anxiety and depression.
  - Query for medication suggestions based on symptoms.

- **Similarity Search**: Implements **Retrieval-Augmented Generation (RAG)** using **Pinecone** vector databases with namespaces to efficiently retrieve and rank relevant information.

- The application is fully containerized using **Docker**, enabling easy deployment across various platforms. It serves as a practical demonstration of how LLM-powered analysis and retrieval systems can be utilized to address mental health challenges.

---


## Technologies Used
- **Data Profiling**: `pandas_profiling`, `y_dataprofiling`
- **EDA and Data Cleaning**: `Python` (`pandas`, `matplotlib`, `seaborn`), `Airflow`, `NLTK`
- **Machine Learning**: `scikit-learn`, `TensorFlow`/`PyTorch`
- **Databases and Storage**: `Snowflake`, `S3`, `Pinecone`
- **Web Application**: `Streamlit`, `Docker`, `AWS`
- **LLM Integration**: `Open AI`, `tiiuae-falcon-7b`
---


