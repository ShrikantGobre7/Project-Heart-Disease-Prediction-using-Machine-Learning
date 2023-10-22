# Project-Heart-Disease-Prediction-using-Machine-Learning

Project Overview
The goal of this project is to build a desktop application that predicts the likelihood of heart disease based on a set of user-input features such as age, gender, blood pressure, cholesterol levels, etc. The prediction will be made using various machine learning algorithms like KNeighborsClassifier, SVC, DecisionTreeClassifier, RandomForestClassifier, and LinearRegression. Users will interact with the application through a user-friendly GUI developed using the tkinter library.

Project Components

Data Collection:
Gather a dataset containing patient information and whether they have heart disease. You can use publicly available datasets like the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

Data Preprocessing:
Load and preprocess the dataset.
Handle missing values, if any.
Encode categorical variables and perform feature scaling.

Machine Learning Models:
Implement the following machine learning models using scikit-learn:
KNeighborsClassifier
SVC (Support Vector Classifier)
DecisionTreeClassifier
RandomForestClassifier
LinearRegression (for regression-based analysis)

Model Training and Evaluation:
Split the dataset into training and testing sets.
Train each model using the training data.
Evaluate the models' performance using appropriate metrics such as accuracy, precision, recall, F1-score, or regression metrics like Mean Absolute Error (MAE) for LinearRegression.

GUI Development:
Use the tkinter library to create a user interface for the application.
Design a form where users can input their information, including age, gender, blood pressure, cholesterol levels, etc.
Implement a button that triggers the prediction based on the selected model.
Display the prediction result on the GUI, indicating whether the user is likely to have heart disease.

Integration:
Integrate the trained machine learning models with the GUI.
Ensure that the input data from the GUI is passed to the selected model for prediction.

User-Friendly Features:
Add clear/reset buttons to allow users to reset the form.
Provide informative labels and error handling for user input.

Tools and Libraries:
Python
scikit-learn for machine learning
tkinter for GUI development
Pandas and NumPy for data manipulation
Matplotlib or Seaborn for data visualization
