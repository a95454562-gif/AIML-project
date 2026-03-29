# AIML Project

## 📌 Objective
To train and evaluate different Machine Learning models using a student dataset.

## 🤖 Models Used
- Linear Regression (for score prediction)
- Logistic Regression (for pass/fail)
- Decision Tree
- Random Forest

## 📊 Dataset
Features:
- Hours_Studied
- Attendance
- Previous_Marks
- Sleep_Hours
- Internet_Usage

Target:
- Final_Score (Regression)
- Pass (Classification)

## ⚙️ How to Run

1. Install libraries:
pip install pandas scikit-learn

2. Run the code:
python model.py

## 📈 Results

- Linear Regression → MSE = 10.5  
- Logistic Regression → Accuracy = 1.0  
- Decision Tree → Accuracy = 1.0  
- Random Forest → Accuracy = 1.0  

## 🧪 Sample Prediction

Input:
[7, 85, 80, 7, 2]

Output:
- Predicted Score: 84.5  
- Prediction: PASS  

## ✅ Conclusion
Random Forest performed best with highest accuracy.

## 📌 Note
The dataset is small, so models may show very high accuracy.
