import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# Step 1: Verify Dataset Exists
dataset_path = 'mobile_usage_behavioral_analysis.csv'
print(f"Checking for dataset at: {os.path.abspath(dataset_path)}")
if not os.path.exists(dataset_path):
    print(f"Error: '{dataset_path}' not found in {os.getcwd()}. Please place it in the project directory.")
    exit(1)

# Load Dataset 
try:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Create target variable (Addiction_Risk) with dynamic bins
try:
    # Use quantiles to ensure balanced classes
    quantiles = df['Total_App_Usage_Hours'].quantile([0.33, 0.66]).values
    bins = [-float('inf'), quantiles[0], quantiles[1], float('inf')]
    print(f"Using bins for Addiction_Risk: {bins}")
    df['Addiction_Risk'] = pd.cut(df['Total_App_Usage_Hours'], 
                                  bins=bins, 
                                  labels=['Low', 'Moderate', 'High'],
                                  include_lowest=True)
    print("Addiction_Risk column created.")
    print("\nAddiction_Risk Distribution:")
    print(df['Addiction_Risk'].value_counts(dropna=False))
except Exception as e:
    print(f"Error creating Addiction_Risk: {e}")
    exit(1)

# Display basic info
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Step 2: Exploratory Data Analysis (EDA)
try:
    print("Starting EDA...")
    plt.figure(figsize=(15, 10))

    # 1. Distribution of Addiction_Risk
    plt.subplot(2, 3, 1)
    sns.countplot(data=df, x='Addiction_Risk')
    plt.title('Distribution of Addiction Risk')
    plt.xticks(rotation=45)
    plt.savefig('eda_risk_distribution.png')
    print("Saved eda_risk_distribution.png")

    # 2. Daily Screen Time vs. Gender
    plt.subplot(2, 3, 2)
    sns.boxplot(data=df, x='Gender', y='Daily_Screen_Time_Hours')
    plt.title('Daily Screen Time by Gender')

    # 3. Correlation Heatmap
    plt.subplot(2, 3, 3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')

    # 4. Social Media Usage vs. Addiction Risk
    plt.subplot(2, 3, 4)
    sns.boxplot(data=df, x='Addiction_Risk', y='Social_Media_Usage_Hours')
    plt.title('Social Media Usage by Addiction Risk')
    plt.xticks(rotation=45)

    # 5. Gaming Usage vs. Age
    plt.subplot(2, 3, 5)
    sns.scatterplot(data=df, x='Age', y='Gaming_App_Usage_Hours', hue='Addiction_Risk')
    plt.title('Gaming Usage vs. Age')

    # 6. App Usage Breakdown by Category
    plt.subplot(2, 3, 6)
    app_usage = df[['Social_Media_Usage_Hours', 'Productivity_App_Usage_Hours', 'Gaming_App_Usage_Hours']].mean()
    sns.barplot(x=app_usage.index, y=app_usage.values)
    plt.title('Average Usage by App Category')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('eda_plots.png')
    print("Saved eda_plots.png")
    plt.show()
except Exception as e:
    print(f"Error in EDA: {e}")
    exit(1)

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Preprocessing
print("Starting preprocessing...")
# Clear duplicates
df = df.drop_duplicates()
print("\nAfter removing duplicates:", df.shape)

# Check for missing classes
class_counts = df['Addiction_Risk'].value_counts()
print("\nClass Distribution After Duplicates:")
print(class_counts)
if len(class_counts) < 3:
    print(f"Warning: Only {len(class_counts)} classes found: {class_counts.index.tolist()}. Adjusting SMOTE.")
    missing_classes = set(['Low', 'Moderate', 'High']) - set(class_counts.index)
    print(f"Missing classes: {missing_classes}")

# Encoding Categorical Variables
try:
    le_gender = LabelEncoder()
    le_location = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    print("Encoded Gender:", df['Gender'].unique())
    df['Location'] = le_location.fit_transform(df['Location'])
    print("Encoded Location:", df['Location'].unique())

    # Encode Addiction_Risk
    le_addiction = LabelEncoder()
    df['Addiction_Risk'] = le_addiction.fit_transform(df['Addiction_Risk'])
    print("Encoded Addiction_Risk:", df['Addiction_Risk'].unique())
except Exception as e:
    print(f"Error in encoding: {e}")
    exit(1)

# Normalization
try:
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['User_ID', 'Addiction_Risk'])
    print("Numeric columns for scaling:", numeric_cols)
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
except Exception as e:
    print(f"Error in normalization: {e}")
    exit(1)

# Display preprocessed data
print("\nPreprocessed Data Head:")
print(df.head())

# Class Imbalance Handling
target_col = 'Addiction_Risk'
print("\nClass Distribution Before Balancing:")
print(pd.Series(df[target_col]).value_counts())

try:
    X = df.drop([target_col, 'User_ID'], axis=1)
    y = df[target_col]
    # Only apply SMOTE if all classes are present
    if len(class_counts) == 3:
        smote = SMOTE(random_state=42, sampling_strategy={0: 400, 1: 400, 2: 400})
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print("\nClass Distribution After SMOTE:")
        print(pd.Series(y_balanced).value_counts())
    else:
        print("Skipping SMOTE due to missing classes. Using original data.")
        X_balanced, y_balanced = X, y
except Exception as e:
    print(f"Error in SMOTE: {e}")
    exit(1)

# Step 4: Model Selection and Training
try:
    print("Starting model training...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
    print("\nTrain Set Shape:", X_train.shape)
    print("Test Set Shape:", X_test.shape)

    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluation
    y_pred = model.predict(X_test)
    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_addiction.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    sns.barplot(x=importances.values, y=importances.index)
    plt.title('Feature Importances for Addiction Prediction')
    plt.savefig('feature_importance.png')
    print("Saved feature_importance.png")
    plt.show()
except Exception as e:
    print(f"Error in model training/evaluation: {e}")
    exit(1)

# Save model and preprocessors
try:
    joblib.dump(model, 'app_addiction_model.pkl')
    joblib.dump(le_gender, 'label_encoder_gender.pkl')
    joblib.dump(le_location, 'label_encoder_location.pkl')
    joblib.dump(le_addiction, 'label_encoder_addiction.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\nTraining complete! Model and preprocessors saved in:", os.getcwd())
except Exception as e:
    print(f"Error saving .pkl files: {e}")
    exit(1)
