# Resume Analysis and Job Role Prediction System

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Data Source](#data-source)
4. [Code Breakdown](#code-breakdown)
   - [Data Loading](#data-loading)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Selection and Training](#model-selection-and-training)
   - [Model Evaluation](#model-evaluation)
   - [Visualization](#visualization)
   - [Prediction Function](#prediction-function)
5. [Results Analysis](#results-analysis)
6. [Usage](#usage)
7. [Customization](#customization)
8. [Troubleshooting](#troubleshooting)

## Project Overview

This project implements a machine learning system to predict the future career paths of computer science students based on their academic performance, skills, and interests. It uses a Random Forest Classifier with Leave-One-Out Cross-Validation (LOOCV) to provide robust predictions even with small datasets.

## Installation

To run this project, you need Python 3.7+ and the following libraries:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Data Source

The project uses a dataset named `cs_students.csv`, which should be placed in the same directory as the script. This dataset contains information about computer science students, including their GPA, age, gender, interests, projects, and programming skills.

## Code Breakdown

### Data Loading

```python
def load_data(self):
    self.df = pd.read_csv(self.file_path)
    print(f"Data loaded. Shape: {self.df.shape}")
```

This function loads the CSV file into a pandas DataFrame.

### Data Preprocessing

```python
def preprocess_data(self):
    # Drop unnecessary columns
    self.df.drop(columns=["Student ID", "Name", "Major"], inplace=True)

    # Encode ordinal categorical variables
    def ordinal_label_encoding(value):
        if value == "Strong":
            return 3
        elif value == "Average":
            return 2
        elif value == "Weak":
            return 1

    ordinal_categorical_features = ["Python", "Java", "SQL"]
    for feature in ordinal_categorical_features:
        self.df[feature] = self.df[feature].apply(ordinal_label_encoding)

    # Encode nominal categorical variables
    nominal_categorical_features = ["Gender", "Interested Domain", "Projects"]
    self.df = pd.get_dummies(self.df, columns=nominal_categorical_features)

    # Encode target variable
    self.df['Future Career'] = self.le.fit_transform(self.df['Future Career'])

    # Prepare features and target
    self.X = self.df.drop('Future Career', axis=1)
    self.y = self.df['Future Career']

    print("Data preprocessed.")
    print(f"Features shape: {self.X.shape}")
    print(f"Target shape: {self.y.shape}")
```

This function preprocesses the data by:
1. Removing unnecessary columns
2. Encoding ordinal categorical variables (Python, Java, SQL skills)
3. One-hot encoding nominal categorical variables (Gender, Interested Domain, Projects)
4. Label encoding the target variable (Future Career)

### Model Selection and Training

```python
def train_and_evaluate_model(self):
    loocv = LeaveOneOut()
    self.model = RandomForestClassifier(random_state=42)

    for train_index, test_index in loocv.split(self.X):
        X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
        y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        self.predictions.extend(y_pred)
        self.true_labels.extend(y_test)

    print("Model training and evaluation completed.")
```

This function uses a Random Forest Classifier with Leave-One-Out Cross-Validation for model training and evaluation.

### Model Evaluation

```python
def calculate_metrics(self):
    accuracy = accuracy_score(self.true_labels, self.predictions)
    precision = precision_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
    recall = recall_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
    f1 = f1_score(self.true_labels, self.predictions, average='weighted', zero_division=0)

    print(f'LOOCV Accuracy: {accuracy:.4f}')
    print(f'LOOCV Precision: {precision:.4f}')
    print(f'LOOCV Recall: {recall:.4f}')
    print(f'LOOCV F1 Score: {f1:.4f}')
```

This function calculates and prints the performance metrics of the model.

### Visualization

```python
def visualize_results(self):
    # Confusion Matrix
    cm = pd.crosstab(pd.Series(self.true_labels, name='Actual'), 
                     pd.Series(self.predictions, name='Predicted'))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': self.X.columns,
        'importance': self.model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.show()
```

This function creates and displays two visualizations:
1. A confusion matrix to show the model's performance across different classes
2. A bar plot of the top 10 feature importances

### Prediction Function

```python
def predict_job_role(self, resume_data):
    # Prepare input data
    input_df = pd.DataFrame([resume_data])
    
    # Encode ordinal features
    for feature in ["Python", "SQL", "Java"]:
        input_df[feature] = input_df[feature].apply(lambda x: 3 if x == "Strong" else (2 if x == "Average" else 1))
    
    # One-hot encode nominal features
    input_df = pd.get_dummies(input_df, columns=["Gender", "Interested Domain", "Projects"])
    
    # Ensure all columns from training are present
    for col in self.X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training data
    input_df = input_df.reindex(columns=self.X.columns, fill_value=0)

    # Make prediction
    prediction = self.model.predict(input_df)
    predicted_job_role = self.le.inverse_transform(prediction)[0]
    
    return predicted_job_role
```

This function takes a new resume as input, preprocesses it in the same way as the training data, and returns a predicted job role.

## Results Analysis

After running the model, we obtained the following results:

```
Data preprocessed.
Model training and evaluation completed.
LOOCV Accuracy: 0.8611
LOOCV Precision: 0.8039
LOOCV Recall: 0.8611
LOOCV F1 Score: 0.8271
```

- **Accuracy (0.8611)**: This indicates that the model correctly predicts the job role 86.11% of the time.
- **Precision (0.8039)**: On average, when the model predicts a specific job role, it is correct 80.39% of the time.
- **Recall (0.8611)**: On average, the model correctly identifies 86.11% of the actual instances of each job role.
- **F1 Score (0.8271)**: This is the harmonic mean of precision and recall, providing a balanced measure of the model's performance.

These results suggest that the model performs well in predicting job roles, with a good balance between precision and recall. The high accuracy indicates that the model is effective in its predictions across all classes.

## Usage

To use this system:

1. Ensure the `cs_students.csv` file is in the same directory as the script.
2. Run the script:
   ```
   python resume_analyzer.py
   ```
3. The system will automatically load the data, preprocess it, train the model, and display results.
4. To predict a job role for a new resume, modify the `new_resume` dictionary in the `main()` function with the desired information.

## Customization

You can customize the system by:
- Modifying the `ordinal_label_encoding()` function to change how skills are encoded.
- Adjusting the `RandomForestClassifier` parameters in `train_and_evaluate_model()` for different model behavior.
- Adding or removing features by modifying the `preprocess_data()` function.

## Troubleshooting

If you encounter issues:
1. Ensure all required libraries are installed.
2. Check that the CSV file is in the correct location and properly formatted.
3. For memory issues with large datasets, consider using batch processing or reducing the number of features.

For any other problems, refer to the error message for specific details.
