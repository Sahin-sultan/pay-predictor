import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load and prepare the Indian salary dataset"""
    print("📊 Loading dataset...")
    df = pd.read_csv('indian_salary_data_500.csv')

    print(f"✅ Dataset loaded successfully!")
    print(f"📈 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")

    return df


def preprocess_data(df):
    """Preprocess the data for training"""
    print("🔄 Preprocessing data...")

    # Check for missing values
    print("🔍 Checking for missing values...")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("⚠️ Missing values found:")
        print(missing_values[missing_values > 0])
        df = df.dropna()
        print("✅ Missing values removed")

    # Remove outliers (salary > 150 lakhs or < 1 lakh)
    initial_count = len(df)
    df = df[(df['salary_inr_lakhs'] >= 1) & (df['salary_inr_lakhs'] <= 150)]
    final_count = len(df)
    print(f"🧹 Removed {initial_count - final_count} outliers")

    # Prepare features and target
    feature_columns = ['age', 'gender', 'education', 'years_of_experience',
                       'job_title', 'job_location', 'city', 'nationality']

    X = df[feature_columns].copy()
    y = df['salary_inr_lakhs'].copy()

    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['gender', 'education',
                           'job_title', 'job_location', 'city', 'nationality']

    for column in categorical_columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
        print(f"✅ Encoded {column}: {len(le.classes_)} unique values")

    return X, y, label_encoders


def train_model(X, y):
    """Train the Gradient Boosting model"""
    print("🤖 Training Gradient Boosting Regressor...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    print(f"📊 Training set size: {len(X_train)}")
    print(f"📊 Test set size: {len(X_test)}")

    # Initialize and train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        verbose=0
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("📈 Model Performance:")
    print(f"   🎯 R² Score: {r2:.4f} ({r2*100:.1f}%)")
    print(f"   📊 MAE: ₹{mae:.2f} Lakhs")
    print(f"   📊 RMSE: ₹{rmse:.2f} Lakhs")

    return model, r2, mae, rmse


def save_model_and_encoders(model, label_encoders):
    """Save the trained model and label encoders"""
    print("💾 Saving model and encoders...")

    joblib.dump(model, 'model.joblib')
    joblib.dump(label_encoders, 'label_encoders.joblib')

    print("✅ Model saved as 'model.joblib'")
    print("✅ Label encoders saved as 'label_encoders.joblib'")


def main():
    print("🚀 Starting Pay Predict Model Training")
    print("=" * 50)

    try:
        # Load and prepare data
        df = load_and_prepare_data()

        # Preprocess data
        X, y, label_encoders = preprocess_data(df)

        # Train model
        model, r2, mae, rmse = train_model(X, y)

        # Save model and encoders
        save_model_and_encoders(model, label_encoders)

        print("=" * 50)
        print("🎉 Training completed successfully!")
        print(f"🎯 Final Model Accuracy: {r2*100:.1f}%")
        print("🚀 Ready to run the Streamlit app!")

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
