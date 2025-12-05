# Titanic_Survival_Prediction
🚢 Titanic Survival Prediction — Machine Learning Project

This project builds a predictive model to determine whether a passenger on the RMS Titanic survived the disaster, based on their demographic and ticket information.
It explores data preprocessing, feature engineering, training multiple ML models, and evaluating them.
A key learning component of this project is demonstrating data leakage and showing its impact compared to a properly cleaned model.

📁 Project Structure
├── Titanic_fixed.ipynb            # Main notebook with full workflow
├── data/                          # Titanic dataset used for training
├── models/
│   ├── model_with_leakage.pkl     # Model trained with leakage
│   ├── model_without_leakage.pkl  # Clean, leakage-free model
└── README.md                      # Project documentation

🎯 Project Objective

To build a machine learning model that predicts:

Survived (1)  or  Did Not Survive (0)


using passenger details such as:

Age

Sex

Ticket Class (Pclass)

Fare

Cabin & Deck info

Family information

Embarked location

This is a binary classification problem.

🛠️ Tech Stack

Python 3

Pandas, NumPy

Matplotlib / Seaborn

Scikit-Learn

XGBoost (optional)

Joblib / Pickle for saving models

🔍 Understanding the Two Models

This project contains two separate models, purposely built to show how data leakage impacts results.

1️⃣ Model WITH Data Leakage
✔ Description

This version mistakenly includes one or more features that should not be used during training because they are not realistically available at prediction time (or give away the target indirectly).

Common leakage sources include:

Using the Survived column during feature engineering

Features derived from post-survival information

Information that is strongly correlated with the target because it relies on future knowledge

🚨 Effect

Training accuracy becomes artificially high

Test accuracy is misleadingly inflated

The model performs unrealistically well on the dataset

The model completely fails in real-world scenarios

This version is included for educational purposes only, to highlight how easy it is to accidentally leak information into a model.

2️⃣ Model WITHOUT Data Leakage
✔ Description

This model is trained using a properly cleaned dataset with:

Correct handling of missing values

Safe categorical encoding

No use of target-related or future-dependent columns

Removal of Cabin, Ticket, Name (or transforming them into non-leaking forms like Deck or Title)

🟢 Effect

Accuracy is realistic

Model generalizes well

Avoids overfitting

Suitable for deployment

This is the correct model and represents the true performance of your learning pipeline.

⚙️ Workflow Summary
✔ Step 1 — Load & Explore Dataset

Basic EDA on age distribution, gender survival rates, fare statistics, etc.

✔ Step 2 — Preprocess Data

Handle missing values

Encode categorical variables

Engineer useful features (FamilySize, Deck, Title)

Train/test split

✔ Step 3 — Train ML Models

Models used typically include:

Logistic Regression

Random Forest

XGBoost (optional)

Both the leaky and clean versions follow this structure for comparison.

✔ Step 4 — Evaluate

Metrics include:

Accuracy

Confusion Matrix

Classification Report

Feature Importance

✔ Step 5 — Save Model

Each trained model is saved in the models/ folder.

📊 Results Summary
Model Type	Expected Accuracy	Notes
With Leakage	⭐ Extremely High (Misleading)	Not realistic; uses invalid information
Without Leakage	✔ 75–85%	True generalization performance
📘 Learning Outcomes

Through this project, you will understand:

🔹 What data leakage is
🔹 How leakage silently boosts accuracy
🔹 How to detect leakage
🔹 How to prevent it
🔹 How proper preprocessing changes model performance

This makes the repository a great educational resource for interviews and ML fundamentals.

🚀 How to Run the Project
git clone <your-repo-url>
cd Titanic-Survival-Prediction

pip install -r requirements.txt

jupyter notebook Titanic_fixed.ipynb
