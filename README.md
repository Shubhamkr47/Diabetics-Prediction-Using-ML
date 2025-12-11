# 🩺 Diabetics Prediction Using Machine Learning

## 🎯 Project Goal
This project aims to develop a machine learning model to accurately predict the onset of diabetes in patients based on various diagnostic measurements. The goal is to provide a reliable predictive tool that can assist in early screening and clinical decision-making.

## 📊 Dataset
The project utilizes the **Pima Indian Diabetes Dataset**, sourced from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).

| Feature | Description |
| :--- | :--- |
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2 hours) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (weight in kg/(height in m)$^2$) |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age (years) |
| **`Outcome`** | Class variable (**0** = Non-diabetic, **1** = Diabetic) |

## 🛠️ Methodology and Pipeline

The project follows a robust Machine Learning pipeline, as detailed in the `Diabetics Prediction Using ML.ipynb` notebook:

1.  **Data Cleaning & Imputation:** Handled critical zero values found in features like `Glucose`, `BloodPressure`, and `BMI` by replacing them with the median of non-zero values.
2.  **Feature Scaling:** Applied `StandardScaler` to normalize the input features.
3.  **Model Comparison:** Evaluated the performance of three popular classification algorithms:
    * **Random Forest Classifier**
    * **Support Vector Machines (SVC)**
    * **Gradient Boosting Classifier (GBC)**
4.  **In-depth Evaluation:** Utilized **Confusion Matrix** and **Classification Report** to assess not just overall accuracy, but also **Precision** (minimizing false diagnoses) and **Recall** (minimizing missed cases of diabetes), which is critical in healthcare.

## ⭐ Key Results

The **Gradient Boosting Classifier (GBC)** delivered the best overall performance in terms of test accuracy and balance across precision and recall.

### Model Accuracy Comparison

| Model | Training Accuracy | Test Accuracy |
| :--- | :--- | :--- |
| **GBC** | ~88.4% | **~78.12%** |
| Random Forest | ~99.4% | ~77.46% |
| SVC | ~77.4% | ~76.19% |

### GBC Classification Report Highlights

The detailed **Classification Report** in the notebook provides the specific Precision, Recall, and F1-scores for both the diabetic and non-diabetic classes, offering a transparent view of the model's predictive power.

## 💻 Dependencies

To run this project, you will need the following Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
