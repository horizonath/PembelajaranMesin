import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
data_file = 'data_balita.csv'
data = pd.read_csv(data_file)

# Display dataset info
st.title("Web Deployment of Decision Tree Model for Balita Data")
st.write("### Dataset Overview")
st.write(data.head())
st.write("Number of rows and columns:", data.shape)

# Feature and target selection
st.write("### Select Features and Target")
features = st.multiselect("Select features:", data.columns.tolist(), default=data.columns[:-1])
target = st.selectbox("Select target:", data.columns.tolist(), index=len(data.columns) - 1)

if features and target:
    X = data[features]
    y = data[target]

    # Encode categorical variables
    encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        encoders[column] = LabelEncoder()
        X[column] = encoders[column].fit_transform(X[column])
    
    # Encode target if categorical
    target_encoder = None
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

    # Determine if regression or classification is required
    is_regression = y.dtype in ['int64', 'float64'] and len(pd.unique(y)) > 10

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the appropriate Decision Tree model
    if is_regression:
        clf = DecisionTreeRegressor()
        st.write("Using DecisionTreeRegressor for regression.")
    else:
        clf = DecisionTreeClassifier()
        st.write("Using DecisionTreeClassifier for classification.")
    clf.fit(X_train, y_train)

    # Display the model performance
    st.write("### Model Performance")
    if is_regression:
        r2_score = clf.score(X_test, y_test)
        st.write(f"R² score on test data: {r2_score:.2f}")
    else:
        accuracy = clf.score(X_test, y_test)
        st.write(f"Accuracy on test data: {accuracy:.2f}")

    # Save the trained model
    model_file = 'decision_tree_balita_model.pkl'
    joblib.dump((clf, encoders, target_encoder), model_file)
    st.write(f"Model saved as {model_file}")

    # Predict section
    st.write("### Predict")
    input_data = {}
    for feature in features:
        if feature in encoders:  # Categorical feature
            options = encoders[feature].classes_
            value = st.selectbox(f"Select {feature}:", options)
            input_data[feature] = [encoders[feature].transform([value])[0]]
        else:  # Numerical feature
            value = st.number_input(f"Input {feature}:", format="%.2f")
            input_data[feature] = [value]

    if st.button("Predict"):
        input_df = pd.DataFrame(input_data)
        prediction = clf.predict(input_df)
        if target_encoder and not is_regression:  # Decode prediction if target was encoded and classification
            predicted_class = target_encoder.inverse_transform(prediction)
            st.write("Prediction:", predicted_class[0])
        else:
            st.write("Prediction:", prediction[0])
