# Titanic-Survival-Prediction
"Machine Learning project to predict Titanic passenger survival using Python,Pandas ,scikit-learn

Project Overview :  

  
The Titanic Survival Prediction project uses Machine Learning to predict whether a passenger survived or not based on various features like age, gender, class, and fare.  
This is a classic binary classification problem and is widely used for beginners to learn data preprocessing, feature engineering, model building, and evaluation.  


Repository Structure : 


Titanic-Survival-Prediction/  
│
├── data/ # Datasets (train.csv, test.csv)    
├── notebooks/ # Jupyter/Colab notebooks with code  
├── src/ # Source code files  
├── images/ # Plots & visualizations  
├── models/ # Saved trained models (optional)    
├── README.md # Project documentation  
└── requirements.txt # Python dependencies  

Problem Statement : 


Predict the survival of Titanic passengers using machine learning models based on features like passenger class, gender, age, siblings, and fare.  

Input: Passenger details (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)    

Output: Survival (0 = No, 1 = Yes)  

📊 Dataset Information : 


The dataset is taken from Kaggle Titanic - Machine Learning from Disaster.  

🔹 Features :  


Feature Description :  

PassengerId Unique ID of a passenger  
Survived Survival (0 = No, 1 = Yes)  
Pclass Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)  
Name Passenger name     
Sex Gender Age Age in  years  
SibSp Number of siblings/spouses aboard  
Parch Number of parents/children aboard  
Ticket Ticket number  
Fare Passenger fare  
Cabin Cabin number  
Embarked Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)  

🛠 Technologies Used : 


Python  
Pandas, NumPy – Data manipulation  
Matplotlib, Seaborn – Data visualization  
Scikit-learn – Model building and evaluation  
XGBoost / Random Forest / Logistic Regression – Classification models  
   
📌 Steps Followed :  


✅ Data Collection – Downloaded dataset from Kaggle  
✅ Data Cleaning – Handling missing values (Age, Cabin, Embarked)     
✅ Feature Engineering – Label Encoding, One-Hot Encoding,scaling  
✅ Exploratory Data Analysis (EDA) – Visualizing survival rates by age, gender, class, etc.  
✅ Model Building – Logistic Regression, Random Forest, XGBoost  
✅ Hyperparameter Tuning – GridSearchCV  
✅ Model Evaluation – Accuracy, Confusion Matrix, Cross-validation  
✅ Prediction on Test Data – Final predictions for submission  

📈 Visualizations :  


🔹 Survival Rate by Gender  
🔹 Survival Rate by Class  

🧠 Machine Learning Models Used : 


Model Accuracy (CV Score) :  

Logistic Regression 81%  
Random Forest 85%  
XGBoost 81%  

📂 How to Run This Project :    

🔹 1️⃣ Clone Repository : 


git clone https://github.com/Puligorladhanush/Titanic-Survival-Prediction.git  
cd Titanic-Survival-Prediction    

🔹 2️⃣ Install Dependencies :  

pip install -r requirements.txt  

🔹 3️⃣ Run the Notebook :  

jupyter notebook notebooks/Titanic_Survival_Prediction.ipynb  

📌 Results & Conclusion :   


The best model achieved ~84% accuracy on the test data.  

Key Factors for Survival: Gender (female more likely), Passenger class (1st class higher chance), Age (children had higher survival).  

📜 Future Improvements :  


🔹 Try deep learning models (ANN)  
🔹 Perform feature selection using PCA  
🔹 Tune hyperparameters for better accuracy    
🔹 Deploy the model using Flask / Streamlit  

📧 Contact :  


👤 Dhanush Puligorla  
📩 Email: dhanushpuligorla@gmail.com  
🌐 GitHub: Puligorladhanush  
