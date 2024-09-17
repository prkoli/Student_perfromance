import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('studentsperformance.csv')

# Inspect data (check column names and data types)
print(data.head())

# Define the features and target variable
# Assuming 'math score', 'reading score', 'writing score' are features and target is one of these scores
data['average_score'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)

# Drop the individual score columns
data = data.drop(['math score', 'reading score', 'writing score'], axis=1)

# Encoding categorical variables
label_encoders = {}
categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target
X = data.drop('average_score', axis=1)  # Features
y = data['average_score']                # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestRegressor()
model.fit(X_train, y_train)

def predict_performance(gender, ethnicity, parental_education, lunch, test_preparation):
    features = pd.DataFrame({
        'gender': [label_encoders['gender'].transform([gender])[0]],
        'race/ethnicity': [label_encoders['race/ethnicity'].transform([ethnicity])[0]],
        'parental level of education': [label_encoders['parental level of education'].transform([parental_education])[0]],
        'lunch': [label_encoders['lunch'].transform([lunch])[0]],
        'test preparation course': [label_encoders['test preparation course'].transform([test_preparation])[0]]
    })
    prediction = model.predict(features)[0]
    return prediction
