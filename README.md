📂 Project Structure
diabetes-prediction-app/
│── app.py                  # Flask backend
│── diabetes_accuracy.py    # Model training and accuracy calculation
│── model.pkl               # Saved ML model
│── diabetes_prediction_dataset.xlsx  # Dataset
│── templates/
│    └── index.html         # Frontend page
│── static/
│    ├── style.css          # Styling
│    └── script.js          # Interactivity
│── README.md               # Project documentation

⚙️ Features

✅ Train a Logistic Regression model on the diabetes dataset

✅ Check accuracy score of the model

✅ User-friendly frontend form for input

✅ Prediction results displayed instantly

✅ Clean design with basic styling

🧑‍💻 Tech Stack

Python (pandas, scikit-learn, Flask)

HTML, CSS, JavaScript (frontend UI)

Git & GitHub (version control and hosting)

🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/prajwalshindhe28/diabetes-prediction-app.git
cd diabetes-prediction-app

2️⃣ Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt


(If requirements.txt doesn’t exist yet, you can create it with:

pip freeze > requirements.txt
```)

### 4️⃣ Run the Flask App
```bash
python app.py


The app will start on 👉 http://127.0.0.1:5000/.

📊 Dataset

The dataset used is diabetes_prediction_dataset.xlsx, which contains medical and lifestyle information such as:

Age

Gender

Smoking History

Hypertension

Heart Disease

HbA1c Level

BMI

Diabetes (target variable)

📈 Model

Algorithm: Logistic Regression

Evaluation Metric: Accuracy Score

Current accuracy: ~XX% (update with your result after training)

🎯 Future Improvements

Add more ML models (Random Forest, XGBoost) for comparison

Deploy on Heroku / Render / Streamlit for public access

Enhance frontend with better UI/UX

Add visualization of model performance (confusion matrix, ROC curve)

🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Prajwal S
