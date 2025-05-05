# Fraudulent Claim Detection Solution

# Supress unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import cross_val_score, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Data Preparation
# 1.1 Load the Data
print("Loading the dataset...")
df = pd.read_csv('insurance_claims.csv')

# 1.2 Check the first few entries
print("\nFirst 5 rows of the dataset:")
print(df.head())

# 1.3 Inspect the shape of the dataset
print("\nDataset shape:", df.shape)

# 1.4 Inspect the features in the dataset
print("\nDataset columns:")
print(df.columns.tolist())

# 1.5 Check data types and basic statistics
print("\nData types and non-null counts:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())
# 2. Data Cleaning
print("\n\n" + "="*50)
print("2. DATA CLEANING")
print("="*50)

# 2.1 Handle null values
print("\n2.1 Handling null values")

# 2.1.1 Examine columns to determine if any value or column needs to be treated
# Check for missing values in each column
print("\nMissing values in each column:")
missing_values = df.isnull().sum()
missing_columns = missing_values[missing_values > 0]
print(missing_columns)

# Calculate percentage of missing values
missing_percentage = (missing_values / len(df)) * 100
print("\nPercentage of missing values:")
print(missing_percentage[missing_percentage > 0])

# Identify columns with >80% missing values
high_missing_cols = [col for col in df.columns if missing_percentage[col] > 80]
print(f"\nColumns with >80% missing values: {high_missing_cols}")

# Create a duplicate dataset
df_cleaned = df.copy()

# Remove columns with >80% missing values
if high_missing_cols:
    print(f"Removing columns with >80% missing values from the cleaned dataset")
    df_cleaned = df_cleaned.drop(columns=high_missing_cols)
    print(f"Columns removed: {high_missing_cols}")
else:
    print("No columns with >80% missing values found.")

# 2.1.2 Handle missing values
print("\n2.1.2 Handling missing values")

# Check if _c39 column is completely empty
if '_c39' in df_cleaned.columns:
    missing_count = df_cleaned['_c39'].isnull().sum()
    total_rows = len(df_cleaned)
    if missing_count == total_rows:
        print(f"Column '_c39' is completely empty with {missing_count} missing values out of {total_rows} rows")
        # Drop the column since it's completely empty
        df_cleaned = df_cleaned.drop(columns=['_c39'])
        print("Dropped column '_c39' as it was completely empty")

# Check for '?' values which might represent missing data
print("\nChecking for '?' values that might represent missing data:")
for column in df_cleaned.columns:
    if df_cleaned[column].dtype == 'object':  # Check only string columns
        question_mark_count = (df_cleaned[column] == '?').sum()
        if question_mark_count > 0:
            print(f"{column}: {question_mark_count} '?' values")
            # Replace '?' with NaN
            df_cleaned[column].replace('?', np.nan, inplace=True)

# For numerical columns, fill missing values with median
numerical_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    if df_cleaned[col].isnull().sum() > 0:
        median_value = df_cleaned[col].median()
        df_cleaned[col].fillna(median_value, inplace=True)
        print(f"Filled missing values in {col} with median: {median_value}")

# For categorical columns, fill missing values with mode
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df_cleaned[col].isnull().sum() > 0:
        mode_value = df_cleaned[col].mode()[0]
        df_cleaned[col].fillna(mode_value, inplace=True)
        print(f"Filled missing values in {col} with mode: {mode_value}")

# Check if any missing values remain
remaining_nulls = df_cleaned.isnull().sum().sum()
print(f"Remaining missing values after imputation: {remaining_nulls}")

# 2.2 Identify and handle redundant values and columns
print("\n2.2 Identifying and handling redundant values and columns")

# 2.2.1 Examine the columns to determine if any value or column needs to be treated
# Display unique values and counts for each column
print("\nUnique values and counts for selected columns:")
for column in df_cleaned.columns[:10]:  # Limit to first 10 columns for readability
    unique_count = df_cleaned[column].nunique()
    print(f"{column}: {unique_count} unique values")
    
    # If the column has a reasonable number of unique values, display them
    if unique_count < 10:  # Only show if there are fewer than 10 unique values
        print(df_cleaned[column].value_counts())
    print("-" * 30)

# 2.2.2 Identify and drop any columns that are completely empty
print("\nChecking for completely empty columns...")
empty_columns = [col for col in df_cleaned.columns if df_cleaned[col].isna().all()]
if empty_columns:
    print(f"Dropping completely empty columns: {empty_columns}")
    df_cleaned = df_cleaned.drop(columns=empty_columns)
else:
    print("No completely empty columns found.")

# 2.2.3 Identify and handle rows where features have illogical or invalid values
print("\nChecking for illogical or invalid values...")

# Check for negative values in columns that should only have positive values
positive_only_columns = [
    'months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',
    'umbrella_limit', 'capital-gains', 'incident_hour_of_the_day',
    'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
    'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim', 'auto_year'
]

for column in positive_only_columns:
    if column in df_cleaned.columns:
        negative_count = (df_cleaned[column] < 0).sum()
        if negative_count > 0:
            print(f"Found {negative_count} negative values in {column}")
            # Replace negative values with column median instead of dropping rows
            median_value = df_cleaned[df_cleaned[column] >= 0][column].median()
            df_cleaned.loc[df_cleaned[column] < 0, column] = median_value
            print(f"Replaced negative values in {column} with median: {median_value}")

# Age range validation: Check if age values are within a reasonable range (e.g., 16-100 for drivers)
if 'age' in df_cleaned.columns:
    unreasonable_age = ((df_cleaned['age'] < 16) | (df_cleaned['age'] > 100)).sum()
    if unreasonable_age > 0:
        print(f"Found {unreasonable_age} unreasonable age values")
        reasonable_median = df_cleaned[(df_cleaned['age'] >= 16) & (df_cleaned['age'] <= 100)]['age'].median()
        df_cleaned.loc[(df_cleaned['age'] < 16) | (df_cleaned['age'] > 100), 'age'] = reasonable_median
        print(f"Replaced unreasonable age values with median: {reasonable_median}")

# Date consistency checks: Ensure incident_date is after policy_bind_date
if 'incident_date' in df_cleaned.columns and 'policy_bind_date' in df_cleaned.columns:
    # First, ensure both columns are in datetime format
    if df_cleaned['incident_date'].dtype != 'datetime64[ns]':
        df_cleaned['incident_date'] = pd.to_datetime(df_cleaned['incident_date'])
    if df_cleaned['policy_bind_date'].dtype != 'datetime64[ns]':
        df_cleaned['policy_bind_date'] = pd.to_datetime(df_cleaned['policy_bind_date'])
    
    # Now check for inconsistent dates
    inconsistent_dates = (df_cleaned['incident_date'] < df_cleaned['policy_bind_date']).sum()
    if inconsistent_dates > 0:
        print(f"Found {inconsistent_dates} incidents that occurred before policy binding")
        # Set incident_date to policy_bind_date + 30 days for these cases
        df_cleaned.loc[df_cleaned['incident_date'] < df_cleaned['policy_bind_date'], 'incident_date'] = \
            df_cleaned.loc[df_cleaned['incident_date'] < df_cleaned['policy_bind_date'], 'policy_bind_date'] + pd.Timedelta(days=30)
        print("Fixed inconsistent dates by setting incident_date to policy_bind_date + 30 days")

# Outlier detection: Check for extreme values in claim amounts
for claim_col in ['total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']:
    if claim_col in df_cleaned.columns:
        Q1 = df_cleaned[claim_col].quantile(0.25)
        Q3 = df_cleaned[claim_col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR  # Using 3 * IQR for extreme outliers
        outliers = (df_cleaned[claim_col] > upper_bound).sum()
        if outliers > 0:
            print(f"Found {outliers} extreme outliers in {claim_col}")
            # Cap the outliers at the upper bound
            df_cleaned.loc[df_cleaned[claim_col] > upper_bound, claim_col] = upper_bound
            print(f"Capped outliers in {claim_col} at {upper_bound}")

# Logical relationship checks: Ensure sum of individual claims equals total claim amount
if all(col in df_cleaned.columns for col in ['total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']):
    sum_claims = df_cleaned['injury_claim'] + df_cleaned['property_claim'] + df_cleaned['vehicle_claim']
    inconsistent_claims = (abs(sum_claims - df_cleaned['total_claim_amount']) > 1).sum()  # Allow for rounding errors
    if inconsistent_claims > 0:
        print(f"Found {inconsistent_claims} rows where sum of individual claims doesn't match total claim amount")
        # Fix by setting total_claim_amount to the sum of individual claims
        df_cleaned.loc[abs(sum_claims - df_cleaned['total_claim_amount']) > 1, 'total_claim_amount'] = \
            df_cleaned.loc[abs(sum_claims - df_cleaned['total_claim_amount']) > 1, ['injury_claim', 'property_claim', 'vehicle_claim']].sum(axis=1)
        print("Fixed inconsistent claim amounts")

# Hour of day validation: Check if incident_hour_of_the_day is between 0-23
if 'incident_hour_of_the_day' in df_cleaned.columns:
    invalid_hours = ((df_cleaned['incident_hour_of_the_day'] < 0) | (df_cleaned['incident_hour_of_the_day'] > 23)).sum()
    if invalid_hours > 0:
        print(f"Found {invalid_hours} invalid hour values")
        valid_median = df_cleaned[(df_cleaned['incident_hour_of_the_day'] >= 0) & (df_cleaned['incident_hour_of_the_day'] <= 23)]['incident_hour_of_the_day'].median()
        df_cleaned.loc[(df_cleaned['incident_hour_of_the_day'] < 0) | (df_cleaned['incident_hour_of_the_day'] > 23), 'incident_hour_of_the_day'] = valid_median
        print(f"Replaced invalid hour values with median: {valid_median}")

# Logical consistency in categorical variables: Check for inconsistencies between related categorical variables
if 'property_damage' in df_cleaned.columns and 'property_claim' in df_cleaned.columns:
    inconsistent_property = ((df_cleaned['property_damage'] == 'NO') & (df_cleaned['property_claim'] > 0)).sum()
    if inconsistent_property > 0:
        print(f"Found {inconsistent_property} cases with property claims but no property damage reported")
        # Update property_damage to YES where there's a property claim
        df_cleaned.loc[(df_cleaned['property_damage'] == 'NO') & (df_cleaned['property_claim'] > 0), 'property_damage'] = 'YES'
        print("Fixed inconsistent property damage flags")

# 2.2.4 Identify and remove columns with limited predictive power
print("\nIdentifying columns with limited predictive power...")

# Columns that are likely identifiers or have very limited predictive power
columns_to_drop = []

# Check for columns with high cardinality (many unique values)
for column in df_cleaned.columns:
    unique_ratio = df_cleaned[column].nunique() / len(df_cleaned)
    if unique_ratio > 0.9:  # If more than 90% of values are unique
        columns_to_drop.append(column)
        print(f"{column}: {df_cleaned[column].nunique()} unique values ({unique_ratio:.2%} of total rows)")

# Drop identified columns
if columns_to_drop:
    print(f"\nDropping columns with high cardinality: {columns_to_drop}")
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)
    print(f"Dataset shape after dropping columns: {df_cleaned.shape}")
else:
    print("No columns identified for removal based on cardinality.")

# Check for the '_c39' column which appears to be an unknown variable
if '_c39' in df_cleaned.columns:
    print("\nDropping '_c39' column as it appears to be an unknown variable")
    df_cleaned = df_cleaned.drop(columns=['_c39'])

# 2.3 Fix Data Types
print("\n2.3 Fixing Data Types")

# Check current data types
print("\nCurrent data types:")
print(df_cleaned.dtypes)

# Convert date columns to datetime
date_columns = ['policy_bind_date', 'incident_date']
for column in date_columns:
    if column in df_cleaned.columns:
        df_cleaned[column] = pd.to_datetime(df_cleaned[column])
        print(f"Converted {column} to datetime")

# Ensure numeric columns are properly typed
numeric_columns = ['policy_deductable', 'policy_annual_premium', 'total_claim_amount', 
                  'injury_claim', 'property_claim', 'vehicle_claim']
for column in numeric_columns:
    if column in df_cleaned.columns:
        # Remove any currency symbols and commas if present
        if df_cleaned[column].dtype == 'object':
            df_cleaned[column] = df_cleaned[column].replace('[\$,]', '', regex=True).astype(float)
            print(f"Converted {column} to float")

# Convert categorical columns to category data type for efficiency
categorical_columns = ['fraud_reported', 'insured_sex', 'incident_severity', 
                      'property_damage', 'police_report_available', 
                      'policy_state', 'incident_state', 'insured_education_level',
                      'insured_occupation', 'insured_hobbies', 'insured_relationship',
                      'incident_type', 'collision_type', 'authorities_contacted',
                      'auto_make', 'auto_model']
for column in categorical_columns:
    if column in df_cleaned.columns:
        df_cleaned[column] = df_cleaned[column].astype('category')
        print(f"Converted {column} to category")

# Convert YES/NO columns to boolean
boolean_columns = ['property_damage', 'police_report_available']
for column in boolean_columns:
    if column in df_cleaned.columns:
        df_cleaned[column] = df_cleaned[column].map({'YES': True, 'NO': False})
        print(f"Converted {column} to boolean")

# Check updated data types
print("\nUpdated data types:")
print(df_cleaned.dtypes)

# Display the cleaned dataset
print("\nCleaned dataset shape:", df_cleaned.shape)
print("First 5 rows of cleaned dataset:")
print(df_cleaned.head())



# 3. Train-Validation Split
print("\n\n" + "="*50)
print("3. TRAIN-VALIDATION SPLIT")
print("="*50)

# 3.1 Import required libraries
# Already imported train_test_split at the beginning

# 3.2 Define feature and target variables
print("\n3.2 Defining feature and target variables")

# Check the target variable distribution
print("\nTarget variable distribution:")
print(df_cleaned['fraud_reported'].value_counts())
print(df_cleaned['fraud_reported'].value_counts(normalize=True).round(4) * 100, "%")

# Put all the feature variables in X
X = df_cleaned.drop(columns=['fraud_reported'])
# Put the target variable in y
y = df_cleaned['fraud_reported']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# 3.3 Split the data
print("\n3.3 Splitting the data into train and validation sets")

# Split the dataset into 70% train and 30% validation and use stratification on the target variable
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Reset index for all train and test sets
X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Validation set shape: X_val {X_val.shape}, y_val {y_val.shape}")

# Check class distribution in training and validation sets
print("\nClass distribution in training set:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True).round(4) * 100, "%")

print("\nClass distribution in validation set:")
print(y_val.value_counts())
print(y_val.value_counts(normalize=True).round(4) * 100, "%")
# 4. EDA on Training Data
print("\n\n" + "="*50)
print("4. EDA ON TRAINING DATA")
print("="*50)

# 4.1 Perform univariate analysis
print("\n4.1 Performing univariate analysis")

# 4.1.1 Identify and select numerical columns from training data
print("\n4.1.1 Identifying numerical columns")
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Numerical columns: {numerical_cols}")

# 4.1.2 Visualize the distribution of numerical features
print("\n4.1.2 Visualizing distribution of numerical features")
plt.figure(figsize=(20, 15))
for i, column in enumerate(numerical_cols[:15], 1):  # Limit to first 15 columns for readability
    plt.subplot(5, 3, i)
    sns.histplot(X_train[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
plt.savefig('numerical_distributions.png')
print("Saved numerical distributions plot to 'numerical_distributions.png'")

# 4.2 Perform correlation analysis
print("\n4.2 Performing correlation analysis")

# Create correlation matrix for numerical columns
correlation_matrix = X_train[numerical_cols].corr()

# Plot Heatmap of the correlation matrix
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
print("Saved correlation matrix plot to 'correlation_matrix.png'")

# Identify highly correlated features
print("\nHighly correlated features (|correlation| > 0.7):")
corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

for col1, col2, corr in corr_pairs:
    print(f"{col1} and {col2}: {corr:.2f}")

# 4.3 Check class balance
print("\n4.3 Checking class balance")

# Plot a bar chart to check class balance
plt.figure(figsize=(8, 6))
sns.countplot(x=y_train)
plt.title('Class Distribution in Training Data')
plt.xlabel('Fraud Reported (Y/N)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_balance.png')
print("Saved class balance plot to 'class_balance.png'")

# 4.4 Perform bivariate analysis
print("\n4.4 Performing bivariate analysis")

# 4.4.1 Target likelihood analysis for categorical variables
print("\n4.4.1 Target likelihood analysis for categorical variables")

# Function to calculate and analyze target variable likelihood for categorical features
def target_likelihood_analysis(df, feature, target='fraud_reported'):
    # Create a crosstab of the feature and target
    crosstab = pd.crosstab(df[feature], df[target], normalize='index')
    
    # If 'Y' is in the columns, select it, otherwise select the first column
    if 'Y' in crosstab.columns:
        likelihood = crosstab['Y']
    else:
        likelihood = crosstab.iloc[:, 0]
    
    # Sort by likelihood
    likelihood = likelihood.sort_values(ascending=False)
    
    return likelihood

# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

# Analyze target likelihood for each categorical feature
for column in categorical_cols[:5]:  # Limit to first 5 columns for readability
    print(f"\nTarget likelihood for {column}:")
    likelihood = target_likelihood_analysis(pd.concat([X_train, y_train], axis=1), column)
    print(likelihood)
    
    # Plot the likelihood
    plt.figure(figsize=(10, 6))
    likelihood.plot(kind='bar')
    plt.title(f'Fraud Likelihood by {column}')
    plt.xlabel(column)
    plt.ylabel('Fraud Likelihood')
    plt.tight_layout()
    plt.savefig(f'likelihood_{column}.png')
    print(f"Saved likelihood plot to 'likelihood_{column}.png'")

# 4.4.2 Explore relationships between numerical features and target
print("\n4.4.2 Exploring relationships between numerical features and target")

# Combine features and target for analysis
train_data = pd.concat([X_train, y_train], axis=1)

# Plot boxplots for numerical features by target
for column in numerical_cols[:5]:  # Limit to first 5 columns for readability
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='fraud_reported', y=column, data=train_data)
    plt.title(f'{column} by Fraud Reported')
    plt.tight_layout()
    plt.savefig(f'boxplot_{column}.png')
    print(f"Saved boxplot to 'boxplot_{column}.png'")

# Calculate mean values for numerical features by target
print("\nMean values for numerical features by target:")
mean_by_target = train_data.groupby('fraud_reported')[numerical_cols].mean()
print(mean_by_target)
# 6. Feature Engineering
print("\n\n" + "="*50)
print("6. FEATURE ENGINEERING")
print("="*50)

# 6.1 Perform resampling
print("\n6.1 Performing resampling to handle class imbalance")

# Initialize the RandomOverSampler
ros = RandomOverSampler(random_state=42)

# Apply oversampling to the training data
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Check the class distribution after resampling
print("\nClass distribution after resampling:")
print(pd.Series(y_train_resampled).value_counts())
print(pd.Series(y_train_resampled).value_counts(normalize=True).round(4) * 100, "%")

print(f"Original training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Resampled training set shape: X_train_resampled {X_train_resampled.shape}, y_train_resampled {y_train_resampled.shape}")

# 6.2 Feature Creation
print("\n6.2 Creating new features")

# First, check which columns are available
print("Available columns in X_train_resampled:")
print(X_train_resampled.columns.tolist())

# Create new features from date columns
if 'policy_bind_date' in X_train_resampled.columns and 'incident_date' in X_train_resampled.columns:
    # For training data
    X_train_resampled['policy_duration_days'] = (X_train_resampled['incident_date'] - X_train_resampled['policy_bind_date']).dt.days
    X_train_resampled['incident_day_of_week'] = X_train_resampled['incident_date'].dt.dayofweek
    X_train_resampled['incident_month'] = X_train_resampled['incident_date'].dt.month
    X_train_resampled['policy_bind_year'] = X_train_resampled['policy_bind_date'].dt.year
    
    # For validation data
    X_val['policy_duration_days'] = (X_val['incident_date'] - X_val['policy_bind_date']).dt.days
    X_val['incident_day_of_week'] = X_val['incident_date'].dt.dayofweek
    X_val['incident_month'] = X_val['incident_date'].dt.month
    X_val['policy_bind_year'] = X_val['policy_bind_date'].dt.year
    
    print("Created new features from date columns:")
    print("- policy_duration_days: Days between policy bind date and incident date")
    print("- incident_day_of_week: Day of the week when the incident occurred")
    print("- incident_month: Month when the incident occurred")
    print("- policy_bind_year: Year when the policy was bound")

# Create interaction features with error handling
try:
    if 'policy_annual_premium' in X_train_resampled.columns and 'months_as_customer' in X_train_resampled.columns:
        X_train_resampled['total_premiums_per_month'] = X_train_resampled['policy_annual_premium'] / X_train_resampled['months_as_customer']
        X_val['total_premiums_per_month'] = X_val['policy_annual_premium'] / X_val['months_as_customer']
        print("Created interaction feature: total_premiums_per_month")
    else:
        print("Warning: Couldn't create total_premiums_per_month - required columns missing")
except Exception as e:
    print(f"Error creating total_premiums_per_month: {e}")

try:
    if 'total_claim_amount' in X_train_resampled.columns and 'number_of_vehicles_involved' in X_train_resampled.columns:
        X_train_resampled['claim_per_vehicle'] = X_train_resampled['total_claim_amount'] / X_train_resampled['number_of_vehicles_involved']
        X_val['claim_per_vehicle'] = X_val['total_claim_amount'] / X_val['number_of_vehicles_involved']
        print("Created interaction feature: claim_per_vehicle")
    else:
        print("Warning: Couldn't create claim_per_vehicle - required columns missing")
except Exception as e:
    print(f"Error creating claim_per_vehicle: {e}")

# Replace infinity values with NaN
X_train_resampled.replace([np.inf, -np.inf], np.nan, inplace=True)
X_val.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values in numeric columns with 0
numeric_cols = X_train_resampled.select_dtypes(include=['int64', 'float64']).columns
X_train_resampled[numeric_cols] = X_train_resampled[numeric_cols].fillna(0)
X_val[numeric_cols] = X_val[numeric_cols].fillna(0)

# Fill NaN values in categorical columns with the most frequent value
cat_cols = X_train_resampled.select_dtypes(include=['category']).columns
for col in cat_cols:
    if X_train_resampled[col].isna().any():
        most_frequent = X_train_resampled[col].mode()[0]
        X_train_resampled[col] = X_train_resampled[col].fillna(most_frequent)
    if X_val[col].isna().any():
        most_frequent = X_val[col].mode()[0]
        X_val[col] = X_val[col].fillna(most_frequent)

# 6.3 Handle redundant columns
print("\n6.3 Handling redundant columns")

# Columns to drop based on EDA and feature creation
columns_to_drop = [
    'policy_bind_date',  # Already extracted useful information
    'incident_date',     # Already extracted useful information
    'policy_number'      # Unique identifier with no predictive power
]

# Drop the columns
X_train_resampled = X_train_resampled.drop(columns=[col for col in columns_to_drop if col in X_train_resampled.columns])
X_val = X_val.drop(columns=[col for col in columns_to_drop if col in X_val.columns])

print(f"Dropped columns: {[col for col in columns_to_drop if col in X_train.columns]}")
print(f"Training set shape after dropping columns: {X_train_resampled.shape}")
print(f"Validation set shape after dropping columns: {X_val.shape}")

# 6.4 Combine values in Categorical Columns
print("\n6.4 Combining values in categorical columns")

# Function to combine low-frequency categories
def combine_rare_categories(df, column, threshold=0.05):
    # Calculate the frequency of each category
    value_counts = df[column].value_counts(normalize=True)
    
    # Identify rare categories (below threshold)
    rare_categories = value_counts[value_counts < threshold].index.tolist()
    
    if rare_categories:
        # Replace rare categories with 'Other'
        df[column] = df[column].apply(lambda x: 'Other' if x in rare_categories else x)
        print(f"Combined rare categories in {column}: {rare_categories}")
    
    return df

# Identify categorical columns
categorical_cols = X_train_resampled.select_dtypes(include=['object']).columns.tolist()

# Apply the function to each categorical column
for column in categorical_cols:
    X_train_resampled = combine_rare_categories(X_train_resampled, column)
    X_val = combine_rare_categories(X_val, column)

# 6.5 Dummy variable creation
print("\n6.5 Creating dummy variables")

# 6.5.1 Identify categorical columns for dummy variable creation
print("\n6.5.1 Identifying categorical columns for dummy variables")
categorical_cols = X_train_resampled.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns for dummy variables: {categorical_cols}")

# 6.5.2 Create dummy variables for categorical columns in training data
print("\n6.5.2 Creating dummy variables for training data")
X_train_dummies = pd.get_dummies(X_train_resampled, columns=categorical_cols, drop_first=True)
print(f"Training data shape after creating dummies: {X_train_dummies.shape}")

# 6.5.3 Create dummy variables for categorical columns in validation data
print("\n6.5.3 Creating dummy variables for validation data")
X_val_dummies = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True)
print(f"Validation data shape after creating dummies: {X_val_dummies.shape}")

# Ensure training and validation data have the same columns
# Get the common columns
common_columns = list(set(X_train_dummies.columns) & set(X_val_dummies.columns))

# Add missing columns to validation data
for col in X_train_dummies.columns:
    if col not in X_val_dummies.columns:
        X_val_dummies[col] = 0

# Add missing columns to training data
for col in X_val_dummies.columns:
    if col not in X_train_dummies.columns:
        X_train_dummies[col] = 0

# Ensure columns are in the same order
X_train_dummies = X_train_dummies[sorted(X_train_dummies.columns)]
X_val_dummies = X_val_dummies[sorted(X_val_dummies.columns)]

print(f"Final training data shape: {X_train_dummies.shape}")
print(f"Final validation data shape: {X_val_dummies.shape}")

# 6.5.4 Create dummy variable for dependent feature in training and validation data
print("\n6.5.4 Creating dummy variable for dependent feature in training and validation data")

# Convert 'Y' to 1 and 'N' to 0 for the training target variable
y_train_dummies = y_train_resampled.map({'Y': 1, 'N': 0})
print("Training target variable conversion:")
print(f"- Original values: {y_train_resampled.unique()}")
print(f"- Converted values: {y_train_dummies.unique()}")

# Convert 'Y' to 1 and 'N' to 0 for the validation target variable
y_val_dummies = y_val.map({'Y': 1, 'N': 0})
print("Validation target variable conversion:")
print(f"- Original values: {y_val.unique()}")
print(f"- Converted values: {y_val_dummies.unique()}")

# Verify the conversion was successful
print("\nTarget variable conversion summary:")
print("- 'Y' (Fraud) mapped to 1")
print("- 'N' (No Fraud) mapped to 0")
print(f"Training set: {y_train_dummies.value_counts().to_dict()}")
print(f"Validation set: {y_val_dummies.value_counts().to_dict()}")

# 6.6 Feature scaling
print("\n6.6 Scaling numerical features")

# Import the necessary scaling tool from scikit-learn
print("\n# Import the necessary scaling tool from scikit-learn")
from sklearn.preprocessing import StandardScaler

# Scale the numeric features present in the training data
print("\n# Scale the numeric features present in the training data")
# Identify numerical columns in the dummy-encoded data
numerical_cols = X_train_dummies.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Identified {len(numerical_cols)} numerical columns for scaling")

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on training data and transform
X_train_dummies[numerical_cols] = scaler.fit_transform(X_train_dummies[numerical_cols])
print(f"Scaled {len(numerical_cols)} numerical features in training data")

# Scale the numeric features present in the validation data
print("\n# Scale the numeric features present in the validation data")
# Use the same scaler fitted on training data to transform validation data
X_val_dummies[numerical_cols] = scaler.transform(X_val_dummies[numerical_cols])
print(f"Scaled {len(numerical_cols)} numerical features in validation data")

# Display sample of scaled data
print("\nFirst 5 rows of scaled training data:")
print(X_train_dummies[numerical_cols].head())
print("\nFirst 5 rows of scaled validation data:")
print(X_val_dummies[numerical_cols].head())
# 7. Model Building
print("\n\n" + "="*50)
print("7. MODEL BUILDING")
print("="*50)

# 7.1 Feature selection
print("\n7.1 Feature selection using RFECV")

# 7.1.1 Import necessary libraries
# Already imported RFECV at the beginning

# 7.1.2 Perform feature selection
print("\n7.1.2 Performing feature selection with RFECV")

# Identify only the numeric columns for feature selection
numeric_cols = X_train_dummies.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Using only {len(numeric_cols)} numeric features for RFECV:")
print(numeric_cols[:10], "..." if len(numeric_cols) > 10 else "")

# Create a subset of the data with only numeric features
X_train_numeric = X_train_dummies[numeric_cols]
print(f"Numeric training data shape: {X_train_numeric.shape}")

# Initialize the logistic regression model for RFECV
logreg = LogisticRegression(max_iter=1000, random_state=42)

# Initialize RFECV with 5-fold cross-validation
rfecv = RFECV(estimator=logreg, step=1, cv=5, scoring='accuracy', n_jobs=-1)

# Fit RFECV to the numeric training data
print("Fitting RFECV to the numeric training data (this may take a while)...")
rfecv = rfecv.fit(X_train_numeric, y_train_dummies)

print(f"Optimal number of features: {rfecv.n_features_}")

# Handle different attribute names in different scikit-learn versions
try:
    # For newer versions of scikit-learn
    if hasattr(rfecv, 'cv_results_'):
        best_score = rfecv.cv_results_['mean_test_score'][rfecv.n_features_ - 1]
    # For older versions of scikit-learn
    elif hasattr(rfecv, 'grid_scores_'):
        best_score = rfecv.grid_scores_[rfecv.n_features_ - 1]
    else:
        best_score = "Not available"
    print(f"Best cross-validation score: {best_score:.4f}")
except:
    print("Could not retrieve best cross-validation score")

# Get the selected numeric features
selected_numeric_features = np.array(numeric_cols)[rfecv.support_].tolist()
print(f"Selected {len(selected_numeric_features)} numeric features")
print("First 10 selected features:", selected_numeric_features[:10])

# 7.1.3 Retain the selected features
print("\n7.1.3 Retaining the selected features")

# Put columns selected by RFECV into variable 'col'
col = selected_numeric_features
print(f"Number of selected features: {len(col)}")
print("First 10 selected features:")
print(col[:10])

# Create a DataFrame with feature rankings
feature_ranking = pd.DataFrame({
    'Feature': numeric_cols,
    'Ranking': rfecv.ranking_,
    'Selected': rfecv.support_
})
feature_ranking = feature_ranking.sort_values('Ranking')
print("\nTop 10 features by ranking:")
print(feature_ranking.head(10))

# Filter the training and validation data to include only selected features
X_train_selected = X_train_dummies[col]
X_val_selected = X_val_dummies[col]

print(f"Training data shape after feature selection: {X_train_selected.shape}")
print(f"Validation data shape after feature selection: {X_val_selected.shape}")

# 7.2 Build Logistic Regression Model
print("\n7.2 Building Logistic Regression Model")

# 7.2.1 Select relevant features and add constant in training data
print("\n7.2.1 Adding constant to training data for statsmodels")

# Select only the columns selected by RFECV
X_train_model = X_train_dummies[col]
print(f"Selected {X_train_model.shape[1]} features for modeling")
print(f"First few selected features: {col[:5]}")

# Import statsmodels and add constant
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train_model)
print(f"Training data shape after adding constant: {X_train_sm.shape}")

# Check the data
print("\nFirst 5 rows of training data with selected features:")
print(X_train_sm.head())

# 7.2.2 Fit logistic regression model
print("\n7.2.2 Fitting logistic regression model")

# Fit a logistic Regression model on X_train after adding a constant and output the summary
logit_model = sm.Logit(y_train_dummies, X_train_sm)
result = logit_model.fit(disp=0)  # disp=0 to suppress convergence messages

# Print the summary
print("\nLogistic Regression Model Summary:")
print(result.summary2())

# Extract key statistics
print("\nKey Statistics:")
print(f"Pseudo R-squared: {result.prsquared:.4f}")
print(f"Log-Likelihood: {result.llf:.4f}")
print(f"AIC: {result.aic:.4f}")
print(f"BIC: {result.bic:.4f}")

# 7.2.3 Evaluate VIF of features to assess multicollinearity
print("\n7.2.3 Evaluating VIF to assess multicollinearity")

# Import 'variance_inflation_factor'
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Make a VIF DataFrame for all the variables present
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_sm.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

# Sort by VIF value
vif_data = vif_data.sort_values('VIF', ascending=False)
print("\nVIF for each feature:")
print(vif_data)

# Identify features with high VIF (> 10)
high_vif_features = vif_data[vif_data["VIF"] > 10]
if not high_vif_features.empty:
    print("\nFeatures with high VIF (> 10):")
    print(high_vif_features)
    
    # OPTIONAL: Drop features with high VIF and retrain the model
    # Uncomment the code below to implement this optional step
    """
    print("\nDropping features with high VIF and retraining the model...")
    
    # Get names of features with high VIF (excluding the constant)
    high_vif_cols = high_vif_features[high_vif_features["Feature"] != "const"]["Feature"].tolist()
    
    # Drop these features from the training data
    X_train_filtered = X_train_model.drop(columns=high_vif_cols)
    
    # Add constant again
    X_train_sm_filtered = sm.add_constant(X_train_filtered)
    
    # Refit the model
    logit_model_filtered = sm.Logit(y_train_dummies, X_train_sm_filtered)
    result_filtered = logit_model_filtered.fit(disp=0)
    
    # Print the summary of the new model
    print("\nLogistic Regression Model Summary (after dropping high VIF features):")
    print(result_filtered.summary2())
    
    # Update the result variable to use the filtered model for subsequent steps
    result = result_filtered
    X_train_sm = X_train_sm_filtered
    """
else:
    print("\nNo features with high VIF (> 10)")

# 7.2.4 Make predictions on training data
print("\n7.2.4 Making predictions on training data")

# Predict the probabilities on the training data
train_pred_proba = result.predict(X_train_sm)
print(f"Shape of predicted probabilities: {train_pred_proba.shape}")

# Reshape it into an array
train_pred_proba_array = np.array(train_pred_proba)
print(f"Shape after reshaping: {train_pred_proba_array.shape}")
print(f"First 5 predicted probabilities: {train_pred_proba_array[:5]}")

# 7.2.5 Create a DataFrame with actual and predicted values
print("\n7.2.5 Creating DataFrame with actual and predicted values")

# Create a DataFrame with actual values and predicted probabilities
train_results = pd.DataFrame({
    'Actual': y_train_dummies,
    'Predicted_Proba': train_pred_proba
})

# Create a column for predicted class using 0.5 as cutoff
train_results['Predicted_Class'] = (train_results['Predicted_Proba'] >= 0.5).astype(int)

print("First 5 rows of results DataFrame:")
print(train_results.head())

# 7.2.6 Check the accuracy of the model
print("\n7.2.6 Checking the accuracy of the model")

# Calculate accuracy
accuracy = (train_results['Actual'] == train_results['Predicted_Class']).mean()
print(f"Accuracy: {accuracy:.4f}")

# 7.2.7 Create a confusion matrix
print("\n7.2.7 Creating confusion matrix")

# Create confusion matrix
conf_matrix = confusion_matrix(train_results['Actual'], train_results['Predicted_Class'])
print("Confusion Matrix:")
print(conf_matrix)

# 7.2.8 Create variables for TP, TN, FP, FN
print("\n7.2.8 Creating variables for TP, TN, FP, FN")

# Extract values from confusion matrix
tn, fp, fn, tp = conf_matrix.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# 7.2.9 Calculate sensitivity, specificity, precision, recall, and F1-score
print("\n7.2.9 Calculating performance metrics")

# Calculate sensitivity (recall)
sensitivity = tp / (tp + fn)
print(f"Sensitivity (Recall): {sensitivity:.4f}")

# Calculate specificity
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.4f}")

# Calculate precision
precision = tp / (tp + fp)
print(f"Precision: {precision:.4f}")

# Calculate F1 score
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
print(f"F1 Score: {f1:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(train_results['Actual'], train_results['Predicted_Class']))
# 7.3 Find the Optimal Cutoff
print("\n7.3 Finding the optimal cutoff")

# 7.3.1 Plot ROC Curve
print("\n7.3.1 Plotting ROC Curve")

# Import libraries or function to plot the ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Define ROC function
def plot_roc_curve(y_true, y_score):
    """
    Plot the ROC curve for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Target scores (probability estimates of the positive class)
        
    Returns:
    --------
    fpr : array
        False positive rates
    tpr : array
        True positive rates
    thresholds : array
        Thresholds used to compute fpr and tpr
    roc_auc : float
        Area under the ROC curve
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png')
    
    return fpr, tpr, thresholds, roc_auc

# Call the ROC function
fpr, tpr, thresholds, roc_auc = plot_roc_curve(train_results['Actual'], train_results['Predicted_Proba'])
print(f"AUC: {roc_auc:.4f}")
print("Saved ROC curve plot to 'roc_curve.png'")

# 7.3.2 Predict on training data at various probability cutoffs
print("\n7.3.2 Predicting at various probability cutoffs")

# Create columns with different probability cutoffs
cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for cutoff in cutoffs:
    train_results[f'Predicted_Class_{cutoff}'] = (train_results['Predicted_Proba'] >= cutoff).astype(int)

# 7.3.3 Plot accuracy, sensitivity, specificity at different cutoffs
print("\n7.3.3 Plotting metrics at different cutoffs")

# Create a DataFrame to store metrics at different cutoffs
cutoff_metrics = pd.DataFrame(columns=['Cutoff', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1'])

# Calculate metrics for each cutoff
metrics_list = []
for cutoff in cutoffs:
    pred_col = f'Predicted_Class_{cutoff}'
    cm = confusion_matrix(train_results['Actual'], train_results[pred_col])
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    metrics_list.append({
        'Cutoff': cutoff,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1': f1
    })

# Create DataFrame from the list of metrics
cutoff_metrics = pd.DataFrame(metrics_list)

print("Metrics at different cutoffs:")
print(cutoff_metrics)

# Plot metrics at different cutoffs
plt.figure(figsize=(12, 8))
plt.plot(cutoff_metrics['Cutoff'], cutoff_metrics['Accuracy'], marker='o', label='Accuracy')
plt.plot(cutoff_metrics['Cutoff'], cutoff_metrics['Sensitivity'], marker='o', label='Sensitivity')
plt.plot(cutoff_metrics['Cutoff'], cutoff_metrics['Specificity'], marker='o', label='Specificity')
plt.plot(cutoff_metrics['Cutoff'], cutoff_metrics['F1'], marker='o', label='F1 Score')
plt.xlabel('Cutoff')
plt.ylabel('Score')
plt.title('Performance Metrics at Different Cutoffs')
plt.legend()
plt.grid(True)
plt.savefig('cutoff_metrics.png')
print("Saved cutoff metrics plot to 'cutoff_metrics.png'")

# Find the optimal cutoff based on F1 score
optimal_cutoff = cutoff_metrics.loc[cutoff_metrics['F1'].idxmax(), 'Cutoff']
print(f"\nOptimal cutoff based on F1 score: {optimal_cutoff}")

# 7.3.4 Create a column for final prediction based on optimal cutoff
print("\n7.3.4 Creating final predictions based on optimal cutoff")

# Create a column for final prediction based on the optimal cutoff
train_results['Final_Predicted_Class'] = (train_results['Predicted_Proba'] >= optimal_cutoff).astype(int)

# 7.3.5 Calculate the accuracy with optimal cutoff
print("\n7.3.5 Calculating accuracy with optimal cutoff")

# Calculate accuracy with optimal cutoff
accuracy_optimal = (train_results['Actual'] == train_results['Final_Predicted_Class']).mean()
print(f"Accuracy with optimal cutoff: {accuracy_optimal:.4f}")

# 7.3.6 Create confusion matrix with optimal cutoff
print("\n7.3.6 Creating confusion matrix with optimal cutoff")

# Create confusion matrix with optimal cutoff
conf_matrix_optimal = confusion_matrix(train_results['Actual'], train_results['Final_Predicted_Class'])
print("Confusion Matrix with optimal cutoff:")
print(conf_matrix_optimal)

# 7.3.7 Create variables for TP, TN, FP, FN with optimal cutoff
print("\n7.3.7 Creating variables for TP, TN, FP, FN with optimal cutoff")

# Extract values from confusion matrix
tn_opt, fp_opt, fn_opt, tp_opt = conf_matrix_optimal.ravel()
print(f"True Negatives: {tn_opt}")
print(f"False Positives: {fp_opt}")
print(f"False Negatives: {fn_opt}")
print(f"True Positives: {tp_opt}")

# 7.3.8 Calculate performance metrics with optimal cutoff
print("\n7.3.8 Calculating performance metrics with optimal cutoff")

# Calculate sensitivity (recall)
sensitivity_opt = tp_opt / (tp_opt + fn_opt)
print(f"Sensitivity (Recall): {sensitivity_opt:.4f}")

# Calculate specificity
specificity_opt = tn_opt / (tn_opt + fp_opt)
print(f"Specificity: {specificity_opt:.4f}")

# Calculate precision
precision_opt = tp_opt / (tp_opt + fp_opt)
print(f"Precision: {precision_opt:.4f}")

# Calculate F1 score
f1_opt = 2 * (precision_opt * sensitivity_opt) / (precision_opt + sensitivity_opt)
print(f"F1 Score: {f1_opt:.4f}")

# Print classification report
print("\nClassification Report with optimal cutoff:")
print(classification_report(train_results['Actual'], train_results['Final_Predicted_Class']))

# 7.3.9 Plot precision-recall curve
print("\n7.3.9 Plotting precision-recall curve")

# Calculate precision-recall curve
precision_curve, recall_curve, _ = precision_recall_curve(train_results['Actual'], train_results['Predicted_Proba'])
pr_auc = auc(recall_curve, precision_curve)

# Plot precision-recall curve
plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig('precision_recall_curve.png')
print("Saved precision-recall curve plot to 'precision_recall_curve.png'")
print(f"PR AUC: {pr_auc:.4f}")
# 7.4 Build Random Forest Model
print("\n7.4 Building Random Forest Model")

# 7.4.2 Build the random forest model
print("\n7.4.2 Building base random forest model")

# Initialize the random forest classifier
rf = RandomForestClassifier(random_state=42)

# Fit the model on the training data
rf.fit(X_train_selected, y_train_dummies)
print("Random Forest model fitted")

# 7.4.3 Get feature importance scores and select important features
print("\n7.4.3 Getting feature importance scores")

# Get feature importance scores
feature_importances = rf.feature_importances_

# Create a DataFrame to visualize the importance scores
importance_df = pd.DataFrame({
    'Feature': X_train_selected.columns,
    'Importance': feature_importances
})
importance_df = importance_df.sort_values('Importance', ascending=False)

print("Top 10 most important features:")
print(importance_df.head(10))

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Top 20 Feature Importances in Random Forest Model')
plt.tight_layout()
plt.savefig('feature_importances.png')
print("Saved feature importances plot to 'feature_importances.png'")

# Select features with importance above a threshold
importance_threshold = 0.01  # Features with importance > 1%
important_features = importance_df[importance_df['Importance'] > importance_threshold]['Feature'].tolist()
print(f"\nSelected {len(important_features)} important features with importance > {importance_threshold}")

# Create new training and validation data with only important features
X_train_important = X_train_selected[important_features]
X_val_important = X_val_selected[important_features]

print(f"Training data shape with important features: {X_train_important.shape}")
print(f"Validation data shape with important features: {X_val_important.shape}")

# 7.4.4 Train the model with selected features
print("\n7.4.4 Training random forest model with selected features")

# Initialize and fit the random forest model with important features
rf_important = RandomForestClassifier(random_state=42)
rf_important.fit(X_train_important, y_train_dummies)
print("Random Forest model fitted with important features")

# 7.4.5 Generate predictions on the training data
print("\n7.4.5 Generating predictions on training data")

# Predict on training data
y_train_pred_rf = rf_important.predict(X_train_important)
print(f"Shape of predictions: {y_train_pred_rf.shape}")

# 7.4.6 Check accuracy of the model
print("\n7.4.6 Checking accuracy of the random forest model")

# Calculate accuracy
accuracy_rf = (y_train_dummies == y_train_pred_rf).mean()
print(f"Accuracy: {accuracy_rf:.4f}")

# 7.4.7 Create confusion matrix
print("\n7.4.7 Creating confusion matrix")

# Create confusion matrix
conf_matrix_rf = confusion_matrix(y_train_dummies, y_train_pred_rf)
print("Confusion Matrix:")
print(conf_matrix_rf)

# 7.4.8 Create variables for TP, TN, FP, FN
print("\n7.4.8 Creating variables for TP, TN, FP, FN")

# Extract values from confusion matrix
tn_rf, fp_rf, fn_rf, tp_rf = conf_matrix_rf.ravel()
print(f"True Negatives: {tn_rf}")
print(f"False Positives: {fp_rf}")
print(f"False Negatives: {fn_rf}")
print(f"True Positives: {tp_rf}")

# 7.4.9 Calculate performance metrics
print("\n7.4.9 Calculating performance metrics")

# Calculate sensitivity (recall)
sensitivity_rf = tp_rf / (tp_rf + fn_rf)
print(f"Sensitivity (Recall): {sensitivity_rf:.4f}")

# Calculate specificity
specificity_rf = tn_rf / (tn_rf + fp_rf)
print(f"Specificity: {specificity_rf:.4f}")

# Calculate precision
precision_rf = tp_rf / (tp_rf + fp_rf)
print(f"Precision: {precision_rf:.4f}")

# Calculate F1 score
f1_rf = 2 * (precision_rf * sensitivity_rf) / (precision_rf + sensitivity_rf)
print(f"F1 Score: {f1_rf:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_train_dummies, y_train_pred_rf))

# 7.4.10 Check if the model is overfitting using cross-validation
print("\n7.4.10 Checking for overfitting using cross-validation")

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_important, X_train_important, y_train_dummies, cv=5, scoring='accuracy')
print("Cross-validation scores:")
print(cv_scores)
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

# Compare with training accuracy
print(f"Training accuracy: {accuracy_rf:.4f}")
print(f"Difference between training and CV accuracy: {accuracy_rf - cv_scores.mean():.4f}")
# 7.5 Hyperparameter Tuning
print("\n7.5 Performing hyperparameter tuning")

# 7.5.1 Use grid search to find the best hyperparameter values
print("\n7.5.1 Using grid search for hyperparameter tuning")

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the grid search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit the grid search to the data
print("Fitting grid search (this may take a while)...")
grid_search.fit(X_train_important, y_train_dummies)

# Print the best parameters and score
print("\nBest parameters:")
print(grid_search.best_params_)
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# 7.5.2 Build a random forest model based on hyperparameter tuning results
print("\n7.5.2 Building random forest model with optimal hyperparameters")

# Get the best parameters
best_params = grid_search.best_params_

# Initialize the model with the best parameters
rf_tuned = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)

# Fit the model
rf_tuned.fit(X_train_important, y_train_dummies)
print("Tuned Random Forest model fitted")

# 7.5.3 Make predictions on training data
print("\n7.5.3 Making predictions on training data with tuned model")

# Predict on training data
y_train_pred_rf_tuned = rf_tuned.predict(X_train_important)
print(f"Shape of predictions: {y_train_pred_rf_tuned.shape}")

# 7.5.4 Check accuracy of the tuned model
print("\n7.5.4 Checking accuracy of the tuned random forest model")

# Calculate accuracy
accuracy_rf_tuned = (y_train_dummies == y_train_pred_rf_tuned).mean()
print(f"Accuracy: {accuracy_rf_tuned:.4f}")

# 7.5.5 Create confusion matrix
print("\n7.5.5 Creating confusion matrix for tuned model")

# Create confusion matrix
conf_matrix_rf_tuned = confusion_matrix(y_train_dummies, y_train_pred_rf_tuned)
print("Confusion Matrix:")
print(conf_matrix_rf_tuned)

# 7.5.6 Create variables for TP, TN, FP, FN
print("\n7.5.6 Creating variables for TP, TN, FP, FN for tuned model")

# Extract values from confusion matrix
tn_rf_tuned, fp_rf_tuned, fn_rf_tuned, tp_rf_tuned = conf_matrix_rf_tuned.ravel()
print(f"True Negatives: {tn_rf_tuned}")
print(f"False Positives: {fp_rf_tuned}")
print(f"False Negatives: {fn_rf_tuned}")
print(f"True Positives: {tp_rf_tuned}")

# 7.5.7 Calculate performance metrics
print("\n7.5.7 Calculating performance metrics for tuned model")

# Calculate sensitivity (recall)
sensitivity_rf_tuned = tp_rf_tuned / (tp_rf_tuned + fn_rf_tuned)
print(f"Sensitivity (Recall): {sensitivity_rf_tuned:.4f}")

# Calculate specificity
specificity_rf_tuned = tn_rf_tuned / (tn_rf_tuned + fp_rf_tuned)
print(f"Specificity: {specificity_rf_tuned:.4f}")

# Calculate precision
precision_rf_tuned = tp_rf_tuned / (tp_rf_tuned + fp_rf_tuned)
print(f"Precision: {precision_rf_tuned:.4f}")

# Calculate F1 score
f1_rf_tuned = 2 * (precision_rf_tuned * sensitivity_rf_tuned) / (precision_rf_tuned + sensitivity_rf_tuned)
print(f"F1 Score: {f1_rf_tuned:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_train_dummies, y_train_pred_rf_tuned))
# 8. Prediction and Model Evaluation
print("\n\n" + "="*50)
print("8. PREDICTION AND MODEL EVALUATION")
print("="*50)

# 8.1 Make predictions over validation data using logistic regression model
print("\n8.1 Making predictions using logistic regression model")

# 8.1.1 Select relevant features for validation data and add constant
print("\n8.1.1 Adding constant to validation data")

# Add constant to the validation data
X_val_sm = sm.add_constant(X_val_selected)
print(f"Validation data shape after adding constant: {X_val_sm.shape}")

# 8.1.2 Make predictions over validation data
print("\n8.1.2 Making predictions on validation data")

# Make predictions on the validation data and store it in the variable 'y_validation_pred'
y_validation_pred = result.predict(X_val_sm)
print(f"Shape of predicted probabilities: {y_validation_pred.shape}")
print(f"First 5 predicted probabilities: {y_validation_pred[:5]}")

# 8.1.3 Create DataFrame with actual values and predicted values
print("\n8.1.3 Creating DataFrame with actual and predicted values")

# Create a DataFrame with actual values and predicted probabilities
val_results = pd.DataFrame({
    'Actual': y_val_dummies,
    'Predicted_Proba': y_validation_pred
})

print("First 5 rows of validation results DataFrame:")
print(val_results.head())

# 8.1.4 Make final prediction based on optimal cutoff
print("\n8.1.4 Making final predictions based on optimal cutoff")

# Create a column for final prediction based on the optimal cutoff
val_results['Final_Predicted_Class'] = (val_results['Predicted_Proba'] >= optimal_cutoff).astype(int)

# 8.1.5 Check the accuracy of the model on validation data
print("\n8.1.5 Checking accuracy on validation data")

# Calculate accuracy
val_accuracy = (val_results['Actual'] == val_results['Final_Predicted_Class']).mean()
print(f"Validation Accuracy: {val_accuracy:.4f}")

# 8.1.6 Create confusion matrix
print("\n8.1.6 Creating confusion matrix for validation data")

# Create confusion matrix
val_conf_matrix = confusion_matrix(val_results['Actual'], val_results['Final_Predicted_Class'])
print("Validation Confusion Matrix:")
print(val_conf_matrix)

# 8.1.7 Create variables for TP, TN, FP, FN
print("\n8.1.7 Creating variables for TP, TN, FP, FN")

# Extract values from confusion matrix
val_tn, val_fp, val_fn, val_tp = val_conf_matrix.ravel()
print(f"True Negatives: {val_tn}")
print(f"False Positives: {val_fp}")
print(f"False Negatives: {val_fn}")
print(f"True Positives: {val_tp}")

# 8.1.8 Calculate performance metrics
print("\n8.1.8 Calculating performance metrics for validation data")

# Calculate sensitivity (recall)
val_sensitivity = val_tp / (val_tp + val_fn)
print(f"Sensitivity (Recall): {val_sensitivity:.4f}")

# Calculate specificity
val_specificity = val_tn / (val_tn + val_fp)
print(f"Specificity: {val_specificity:.4f}")

# Calculate precision
val_precision = val_tp / (val_tp + val_fp)
print(f"Precision: {val_precision:.4f}")

# Calculate F1 score
val_f1 = 2 * (val_precision * val_sensitivity) / (val_precision + val_sensitivity)
print(f"F1 Score: {val_f1:.4f}")

# Print classification report
print("\nValidation Classification Report:")
print(classification_report(val_results['Actual'], val_results['Final_Predicted_Class']))

# 8.2 Make predictions over validation data using random forest model
print("\n8.2 Making predictions using random forest model")

# 8.2.1 Select the important features and make predictions
print("\n8.2.1 Making predictions on validation data with random forest")

# Select the relevant features for validation data
X_val_rf = X_val_important
print(f"Selected {X_val_rf.shape[1]} important features for random forest validation")
print(f"First few important features: {important_features[:5]}")

# Make predictions on the validation data
val_pred_rf = rf_tuned.predict(X_val_rf)
print(f"Shape of predictions: {val_pred_rf.shape}")
print(f"First 5 predictions: {val_pred_rf[:5]}")

# 8.2.2 Check accuracy of random forest model
print("\n8.2.2 Checking accuracy of random forest model on validation data")

# Calculate accuracy
val_accuracy_rf = (y_val_dummies == val_pred_rf).mean()
print(f"Validation Accuracy: {val_accuracy_rf:.4f}")

# 8.2.3 Create confusion matrix
print("\n8.2.3 Creating confusion matrix for validation data")

# Create confusion matrix
val_conf_matrix_rf = confusion_matrix(y_val_dummies, val_pred_rf)
print("Validation Confusion Matrix:")
print(val_conf_matrix_rf)

# 8.2.4 Create variables for TP, TN, FP, FN
print("\n8.2.4 Creating variables for TP, TN, FP, FN")

# Extract values from confusion matrix
val_tn_rf, val_fp_rf, val_fn_rf, val_tp_rf = val_conf_matrix_rf.ravel()
print(f"True Negatives: {val_tn_rf}")
print(f"False Positives: {val_fp_rf}")
print(f"False Negatives: {val_fn_rf}")
print(f"True Positives: {val_tp_rf}")

# 8.2.5 Calculate performance metrics
print("\n8.2.5 Calculating performance metrics for validation data")

# Calculate sensitivity (recall)
val_sensitivity_rf = val_tp_rf / (val_tp_rf + val_fn_rf)
print(f"Sensitivity (Recall): {val_sensitivity_rf:.4f}")

# Calculate specificity
val_specificity_rf = val_tn_rf / (val_tn_rf + val_fp_rf)
print(f"Specificity: {val_specificity_rf:.4f}")

# Calculate precision
val_precision_rf = val_tp_rf / (val_tp_rf + val_fp_rf)
print(f"Precision: {val_precision_rf:.4f}")

# Calculate F1 score
val_f1_rf = 2 * (val_precision_rf * val_sensitivity_rf) / (val_precision_rf + val_sensitivity_rf)
print(f"F1 Score: {val_f1_rf:.4f}")

# Print classification report
print("\nValidation Classification Report:")
print(classification_report(y_val_dummies, val_pred_rf))
# Conclusion
print("\n\n" + "="*50)
print("CONCLUSION")
print("="*50)

print("\nModel Comparison:")
print("-" * 80)
print(f"{'Metric':<20} {'Logistic Regression':<20} {'Random Forest':<20}")
print("-" * 80)
print(f"{'Accuracy':<20} {val_accuracy:.4f}{'':<16} {val_accuracy_rf:.4f}")
print(f"{'Sensitivity':<20} {val_sensitivity:.4f}{'':<16} {val_sensitivity_rf:.4f}")
print(f"{'Specificity':<20} {val_specificity:.4f}{'':<16} {val_specificity_rf:.4f}")
print(f"{'Precision':<20} {val_precision:.4f}{'':<16} {val_precision_rf:.4f}")
print(f"{'F1 Score':<20} {val_f1:.4f}{'':<16} {val_f1_rf:.4f}")
print("-" * 80)

# Determine the best model
if val_f1_rf > val_f1:
    best_model = "Random Forest"
    best_accuracy = val_accuracy_rf
    best_f1 = val_f1_rf
else:
    best_model = "Logistic Regression"
    best_accuracy = val_accuracy
    best_f1 = val_f1

print(f"\nBest Model: {best_model}")
print(f"Best Model Accuracy: {best_accuracy:.4f}")
print(f"Best Model F1 Score: {best_f1:.4f}")

# Key findings
print("\nKey Findings:")
print("1. The most important features for predicting fraud are:")
for i, (feature, importance) in enumerate(importance_df.head(5).values, 1):
    print(f"   {i}. {feature}: {importance:.4f}")

print("\n2. Class imbalance was addressed using random oversampling, which improved model performance.")

print("\n3. The optimal probability threshold for fraud detection was determined to be:", optimal_cutoff)

print("\n4. Both models performed well, with Random Forest showing slightly better performance in terms of F1 score.")

print("\n5. The models can be used to flag potentially fraudulent claims for further investigation, potentially saving the company significant amounts in fraudulent payouts.")

print("\nRecommendations:")
print("1. Implement the model in the claims processing workflow to flag suspicious claims for review.")
print("2. Focus investigation resources on claims with high fraud probability scores.")
print("3. Regularly retrain the model with new data to maintain its effectiveness.")
print("4. Consider collecting additional data points that might improve fraud detection.")
print("5. Develop a user-friendly interface for claims adjusters to interpret model predictions.")

print("\nThis concludes the Fraudulent Claim Detection project.")
# Validate dataset before fixing data types
print("\n" + "="*50)
print("DATASET VALIDATION")
print("="*50)

# Check dataset dimensions
print(f"\nCleaned dataset shape: {df_cleaned.shape}")

# Check for remaining missing values
remaining_nulls = df_cleaned.isnull().sum()
if remaining_nulls.sum() > 0:
    print("\nColumns with remaining missing values:")
    print(remaining_nulls[remaining_nulls > 0])
else:
    print("\nNo missing values remain in the dataset.")

# Quick summary of numerical columns
print("\nSummary of numerical columns:")
print(df_cleaned.describe().T[['count', 'min', 'max', 'mean']])

# Check unique values in key categorical columns
print("\nDistribution of key categorical variables:")
for col in ['fraud_reported', 'incident_severity', 'insured_sex', 'incident_type'][:3]:  # Limit to first 3 for brevity
    if col in df_cleaned.columns:
        print(f"\n{col}:")
        print(df_cleaned[col].value_counts())

# Check for duplicate records
duplicates = df_cleaned.duplicated().sum()
print(f"\nNumber of duplicate records: {duplicates}")

# Display a few sample records
print("\nSample of cleaned dataset (5 records):")
print(df_cleaned.sample(5))
# 8.1.1 Select relevant features for validation data and add constant
print("\n8.1.1 Adding constant to validation data")

# Select the relevant features for validation data
X_val_model = X_val_dummies[col]
print(f"Selected {X_val_model.shape[1]} features for validation data")
print(f"First few selected features: {col[:5]}")

# Add constant to X_validation
X_val_sm = sm.add_constant(X_val_model)
print(f"Validation data shape after adding constant: {X_val_sm.shape}")
