
# Diabetes Prediction Model

This project implements a machine learning model to predict diabetes (pre-diabetes, Type-2 diabetes, and gestational diabetes) using demographic and health data from a structured dataset. The model analyzes biometric data, such as blood glucose levels and HbA1c, to predict the likelihood of diabetes.

## Datasets

The code is sourced from Kaggle with a few modifications for my own datasets that I have collected. The datasets I have collected are under the "Releases" section of my Github page for this project along with the source code

The dataset is sourced from Kaggle and includes features such as gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, and blood glucose level.

- **Dataset link**: [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

### Sample Data

| Gender | Age | Hypertension | Heart Disease | Smoking History | BMI  | HbA1c Level | Blood Glucose Level | Diabetes |
|--------|-----|--------------|---------------|-----------------|------|-------------|---------------------|----------|
| Female | 80  | 0            | 1             | Never           | 25.19| 6.6         | 140                 | 0        |
| Female | 54  | 0            | 0             | No Info         | 27.32| 6.6         | 80                  | 0        |
| Male   | 28  | 0            | 0             | Never           | 27.32| 5.7         | 158                 | 0        |
| Female | 36  | 0            | 0             | Current         | 23.45| 5.0         | 155                 | 0        |

## Models

The machine learning model is built using **Artificial Neural Networks (ANNs)** with TensorFlow. ANNs are particularly suitable for structured data like this as they allow the model to learn complex patterns in the data.

### Key Layers:

- **Input Layer**: Accepts the demographic and biometric data.
- **Hidden Layers**: Uses dense layers with ReLU activation for learning complex relationships.
- **Output Layer**: A binary classification (diabetes vs. non-diabetes) using a sigmoid activation function.
- **Output Layer** (for my datasets): because I am calculating prediction in 3 different classes (prediabetes, type-2 diabetes, and gestational diabetes) I used softmax as the output layer activation function

The model from Kaggle is trained using the Adam optimizer and binary cross-entropy loss, which is appropriate for a binary classification task.
The models that I have built use a different number of techniques with different optimizers (RMSProp) and categorical-cross entropy loss. I have also incorporated ensemble models using RandomForest and XGBoost algorithms.

### Results

The model from Kaggle demonstrates strong predictive power with the following performance:

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

They should be found before each model compilation. So just run those cells.

### 3. Download the Dataset

Download the diabetes dataset from Kaggle [here](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) and place it in the `data/` directory of your project.
The rest of the datasets I have collected are under "Releases". Just download those and the source code I have uploaded.
### 4. Run the Jupyter Notebook

To train the model or run inference, open and execute the notebook:

```bash
jupyter notebook DS_Capstone_Assignment2_1.ipynb
```

## Application to Ongoing Projects

### Real-Time Health Coaching

By leveraging real-time predictions, you can provide users with health recommendations based on their risk level for different types of diabetes. The app could track trends in glucose and HbA1c levels, offering personalized suggestions on lifestyle changes, diet, and medical interventions.

### 3. Custom Dataset Integration

The model can be adapted to include time-series data (e.g., daily glucose readings) from your custom dataset. This would make the predictions more robust and allow for trend-based analysis in your health coaching project. Because the model is eventually gonna predict 3 different classes, it is best to combine information from several different datasets that correspond with each diagnosis.
