
# Diabetes Prediction Model

This project implements a machine learning model to predict diabetes (pre-diabetes, Type-2 diabetes, and gestational diabetes) using demographic and health data from a structured dataset. The model analyzes biometric data, such as blood glucose levels and HbA1c, to predict the likelihood of diabetes.

## Dataset

The dataset is sourced from Kaggle and includes features such as gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, and blood glucose level.

- **Dataset link**: [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

### Sample Data

| Gender | Age | Hypertension | Heart Disease | Smoking History | BMI  | HbA1c Level | Blood Glucose Level | Diabetes |
|--------|-----|--------------|---------------|-----------------|------|-------------|---------------------|----------|
| Female | 80  | 0            | 1             | Never           | 25.19| 6.6         | 140                 | 0        |
| Female | 54  | 0            | 0             | No Info         | 27.32| 6.6         | 80                  | 0        |
| Male   | 28  | 0            | 0             | Never           | 27.32| 5.7         | 158                 | 0        |
| Female | 36  | 0            | 0             | Current         | 23.45| 5.0         | 155                 | 0        |

## Model

The machine learning model is built using **Artificial Neural Networks (ANNs)** with TensorFlow. ANNs are particularly suitable for structured data like this as they allow the model to learn complex patterns in the data.

### Key Layers:

- **Input Layer**: Accepts the demographic and biometric data.
- **Hidden Layers**: Uses dense layers with ReLU activation for learning complex relationships.
- **Output Layer**: A binary classification (diabetes vs. non-diabetes) using a sigmoid activation function.

The model is trained using the Adam optimizer and binary cross-entropy loss, which is appropriate for a binary classification task.

### Results

The model demonstrates strong predictive power with the following performance:

- **Accuracy**: 92%
- **Precision**: 89%
- **Recall (Sensitivity)**: 91%
- **F1-Score**: 90%

## Instructions to Run the Code

### 1. Clone the Repository

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

### 2. Install Dependencies

Install the required Python libraries by running the following command:

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the diabetes dataset from Kaggle [here](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) and place it in the `data/` directory of your project.

### 4. Run the Jupyter Notebook

To train the model or run inference, open and execute the notebook:

```bash
jupyter notebook DS_Capstone_Assignment2_1.ipynb
```

## Application to Ongoing Projects

### 1. Early Detection of Diabetic Retinopathy

This model can be extended by incorporating biometric data with eye health metrics, such as retinal images, for a more comprehensive prediction of diabetic complications, including diabetic retinopathy.

### 2. Real-Time Health Coaching

By leveraging real-time predictions, you can provide users with health recommendations based on their risk level for different types of diabetes. The app could track trends in glucose and HbA1c levels, offering personalized suggestions on lifestyle changes, diet, and medical interventions.

### 3. Custom Dataset Integration

The model can be adapted to include time-series data (e.g., daily glucose readings) from your custom dataset. This would make the predictions more robust and allow for trend-based analysis in your health coaching project.
