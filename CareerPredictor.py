import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


class ResumeAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.le = LabelEncoder()
        self.model = None
        self.predictions = []
        self.true_labels = []

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print(f"Data loaded. Shape: {self.df.shape}")

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
            self.df[feature] = self.df[feature].apply(ordinal_label_encoding).astype(int)

        # Encode nominal categorical variables
        nominal_categorical_features = ["Gender", "Interested Domain", "Projects"]
        self.df = pd.get_dummies(self.df, columns=nominal_categorical_features)

        # Encode target variable
        self.df['Future Career'] = self.le.fit_transform(self.df['Future Career'])

        # Prepare features and target
        self.X = self.df.drop('Future Career', axis=1)
        self.y = self.df['Future Career']

        print("Data preprocessed.")

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

    def calculate_metrics(self):
        accuracy = accuracy_score(self.true_labels, self.predictions)
        precision = precision_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        recall = recall_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        f1 = f1_score(self.true_labels, self.predictions, average='weighted', zero_division=0)

        print(f'LOOCV Accuracy: {accuracy:.4f}')
        print(f'LOOCV Precision: {precision:.4f}')
        print(f'LOOCV Recall: {recall:.4f}')
        print(f'LOOCV F1 Score: {f1:.4f}')

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

    def predict_job_role(self, resume_data):
        # Prepare input data
        input_df = pd.DataFrame([resume_data])
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


def main():
    analyzer = ResumeAnalyzer('cs_students.csv')
    analyzer.load_data()
    analyzer.preprocess_data()
    analyzer.train_and_evaluate_model()
    analyzer.calculate_metrics()
    analyzer.visualize_results()

    # Example: Predict job role for a new resume
    new_resume = {
        'GPA': 3.7,
        'Age': 22,
        'Gender': 'Female',
        'Interested Domain': "Machine Learning",
        'Projects': "Image Recognition",
        'Python': "Strong",
        'SQL': "Average",
        'Java': "Weak"
    }

    predicted_role = analyzer.predict_job_role(new_resume)
    print(f"\nPredicted Job Role: {predicted_role}")


if __name__ == "__main__":
    main()