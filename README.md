# Mental Health Counseling Analysis

This repository contains the implementation of a project analyzing mental health counseling transcripts. The primary goals of this project are to profile and clean the dataset, perform exploratory data analysis (EDA), build predictive machine learning models, and create a web application for practical use. Below is an overview of the repository's contents and functionalities.

---

## Repository Structure

### 1. **Data Profiling**
- Folder: `data_profiling`
- **Description**: This folder contains scripts and notebooks for profiling and understanding the dataset. Python tools like `pandas` and `y_dataprofiling` have been used for detailed profiling to identify key data characteristics, outliers, and potential data quality issues.

### 2. **Data Cleaning and EDA**
- **Description**: Comprehensive data cleaning and exploratory analysis have been performed, including:
  - Handling missing values
  - Removing duplicates
  - Standardizing text and numeric fields
  - Visualizing trends and distributions
- **Tools Used**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `ydata_profiling`
- **Files**: The corresponding notebooks for these steps are available in the repository.

### 3. **Machine Learning Models**
- Folder: `machine_learning_models`
- **Description**: Two machine learning models were built to infer useful insights from the dataset:
  1. **Classification Model 1**: Predicts the mental disorder of a person based on counseling transcripts.
     - Classes:
       - Bipolar Type-2
       - Depression
       - Normal
       - Bipolar Type-1
  2. **Classification Model 2**: Predicts the mood swings of a person.
     - Classes:
       - High
       - Medium
       - Low
- **Files**: The implementation details and training notebooks for these models are included in this folder.

### 4. **Web Application**
- **Description**: A web application integrates one of the following functionalities:
  - Users can enter free-text describing a challenge and receive suggestions on how to address it using a pre-trained large language model (LLM).
- **Deployment**: The application demonstrates practical use cases of the analysis and models.

---

## Technologies Used
- **Data Profiling**: `pandas_profiling`, `y_dataprofiling`
- **EDA and Data Cleaning**: Python (`pandas`, `matplotlib`, `seaborn`)
- **Machine Learning**: `scikit-learn`, `TensorFlow`/`PyTorch`
- **Web Application**: Streamlit
- **LLM Integration**: OpenAI API (within budget constraints)

---

## How to Use
1. **Data Profiling**:
   - Navigate to the `data_profiling` folder and run the provided scripts/notebooks to generate detailed data profiling reports.

2. **Machine Learning Models**:
   - Access the `machine_learning_models` folder for model training and evaluation scripts.
   - Ensure the necessary libraries are installed by referring to `requirements.txt`.

3. **Web Application**:
   - Deploy the web application by running the Streamlit script provided in the repository.

---

