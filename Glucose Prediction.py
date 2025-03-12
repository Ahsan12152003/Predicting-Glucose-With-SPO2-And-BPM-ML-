import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data from the new Excel file
file_path = 'C:\\fyp\\Diabetic + Non Diabetic.xlsx'
data = pd.read_excel(file_path)

# Select the relevant columns
selected_columns = [
    'BMI', 
    'Gender',
    'Age', 
    'NIV  (Tranmittance) (IR)', 
    'NIV (absorbance)(IR)', 
    'NIV (Transmittance)(RED)', 
    'NIV (Absorbance)(RED)', 
    'IV Glucose(mg/dl)'
]

# Extract the relevant columns
data_selected = data[selected_columns]

# Handle missing values by dropping rows with any missing values
data_cleaned = data_selected.dropna()

# Encode the 'Gender' column using .loc to avoid SettingWithCopyWarning
label_encoder = LabelEncoder()
data_cleaned.loc[:, 'Gender'] = label_encoder.fit_transform(data_cleaned['Gender'])

# Add a column to indicate diabetic (1) or non-diabetic (0)
data_cleaned['Diabetic'] = 0
data_cleaned.loc[53:89, 'Diabetic'] = 1

# Define features (X) and target variable (y) for regression
X_reg = data_cleaned[['BMI', 'Gender', 'Age', 'NIV  (Tranmittance) (IR)', 'NIV (absorbance)(IR)', 'NIV (Transmittance)(RED)', 'NIV (Absorbance)(RED)']]
y_reg = data_cleaned['IV Glucose(mg/dl)']

# Split the data into training and testing sets for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data for regression
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Initialize and train the regression model with scaled data
model_reg_scaled = LinearRegression()
model_reg_scaled.fit(X_train_reg_scaled, y_train_reg)

# Make predictions with the scaled regression model
y_pred_reg_scaled = model_reg_scaled.predict(X_test_reg_scaled)

# Evaluate the scaled regression model
mae_scaled = mean_absolute_error(y_test_reg, y_pred_reg_scaled)
mse_scaled = mean_squared_error(y_test_reg, y_pred_reg_scaled)
r2_scaled = r2_score(y_test_reg, y_pred_reg_scaled)

print(f'Regression MAE: {mae_scaled}')
print(f'Regression MSE: {mse_scaled}')
print(f'Regression R2: {r2_scaled}')

# Define features (X) and target variable (y) for classification
X_clf = X_reg
y_clf = data_cleaned['Diabetic']

# Split the data into training and testing sets for classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Transform both training and test data for classification using the same scaler
X_train_clf_scaled = scaler.transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

# Initialize and train the classification model with scaled data
model_clf_scaled = RandomForestClassifier(random_state=42)
model_clf_scaled.fit(X_train_clf_scaled, y_train_clf)

# Make predictions with the scaled classification model
y_pred_clf_scaled = model_clf_scaled.predict(X_test_clf_scaled)

# Evaluate the scaled classification model
accuracy_scaled = accuracy_score(y_test_clf, y_pred_clf_scaled)

print(f'Classification Accuracy: {accuracy_scaled}')

# Function to predict glucose level and diabetes status based on new NIV data
def predict_glucose_and_diabetes(bmi, gender, age, niv_ir_transmittance, niv_ir_absorbance, niv_red_transmittance, niv_red_absorbance):
    # Encode the gender
    gender_encoded = label_encoder.transform([gender])[0]
    
    # Create a DataFrame for the new data
    new_data = pd.DataFrame({
        'BMI': [bmi],
        'Gender': [gender_encoded],
        'Age': [age],
        'NIV  (Tranmittance) (IR)': [niv_ir_transmittance],
        'NIV (absorbance)(IR)': [niv_ir_absorbance],
        'NIV (Transmittance)(RED)': [niv_red_transmittance],
        'NIV (Absorbance)(RED)': [niv_red_absorbance]
    })
    
    # Scale the new data using the previously fitted scaler
    new_data_scaled = scaler.transform(new_data)
    
    # Predict the glucose level using the trained regression model
    glucose_prediction = model_reg_scaled.predict(new_data_scaled)
    
    # Predict the diabetes status using the trained classification model
    diabetes_prediction = model_clf_scaled.predict(new_data_scaled)
    diabetes_status = 'Diabetic' if diabetes_prediction[0] == 1 else 'Non-Diabetic'
    
    return glucose_prediction[0], diabetes_status

# Example usage of the prediction function
bmi = 30
gender = 'Male'
age = 23
niv_ir_transmittance = 12
niv_ir_absorbance = 0.97
niv_red_transmittance = 13
niv_red_absorbance = 0.74

# Predict glucose level and diabetes status
predicted_glucose, diabetes_status = predict_glucose_and_diabetes(bmi, gender, age, niv_ir_transmittance, niv_ir_absorbance, niv_red_transmittance, niv_red_absorbance)

# Print the results
print(f'Predicted Glucose Level: {predicted_glucose}')
print(f'Diabetes Status: {diabetes_status}')