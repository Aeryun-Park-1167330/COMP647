# %%
# COMP647 Assignment 02 - Data Preprocessing
# Supplement Recommendation System Based on Health and Dietary Data
# Student: Aeryun Park | Dataset: Patient Health Information 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', 1000)      
pd.set_option('display.max_rows', None)   
pd.set_option('display.max_colwidth', None) 

# Load the healthcare dataset containing patient information
# This dataset will be used to recommend appropriate Korure supplement categories
# based on individual health conditions and lifestyle factors
df = pd.read_csv('1167330_AeryunPark.csv')

# %%
# Display dataset dimensions to understand the scale of patient data
# Important for understanding the scope of our supplement recommendation model
print ("Dataset Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# %%
# Examine the structure and first few records to understand health data format
# This helps identify key health indicators relevant for supplement classification
df.head()

# %%
df.columns.tolist()

# %%
df.dtypes

# %%
# Get basic statistics
df.describe()

# %%
df.info()

# %%
# Comprehensive missing data assessment across all health variables
# Missing health data patterns can affect supplement recommendation accuracy
df.isnull().sum()

# %%
# Detailed analysis of categorical health columns with missing values
missing_cols = ['Disease_Type', 'Dietary_Restrictions', 'Allergies']
for col in missing_cols:
    print(f"{col} unique value:")
    print(df[col].value_counts(dropna=False))
    print("-"*30)

# %%
# Use domain-specific 'None' imputation for supplement recommendation context
# Missing values in health-related categorical variables likely indicate absence
# of conditions rather than unknown conditions, which is valuable for supplement targeting
df['Disease_Type'] = df['Disease_Type'].fillna('None')
df['Dietary_Restrictions'] = df['Dietary_Restrictions'].fillna('None')
df['Allergies'] = df['Allergies'].fillna('None')

# Verification that imputation was successful
df.isnull().sum()

# %%
# Check Patient_ID duplicates first (most important)
patient_duplicates = df['Patient_ID'].duplicated().sum()
print(f"Patient_ID Duplicate Count: {patient_duplicates}")
print("*" * 50)

# %%
# Optional : Quick check for suspicious duplicates
suspicious_cols = ['Weight_kg', 'Height_cm', 'BMI']
for col in suspicious_cols:
    if col in df.columns:
        dup_count = df[col].duplicated().sum()
        print(f"{col}: {dup_count} duplicates")

# %%
# Check if these duplicates are normal
print("== Duplicate Analysis ==")
total_rows = len(df)
print(f"Total patients: {total_rows}")

cols_to_check = ['Weight_kg', 'Height_cm', 'BMI']
for col in cols_to_check:
    if col in df.columns:
        dup_count = df[col].duplicated().sum()
        percentage = (dup_count / total_rows) * 100
        print(f"{col}: {dup_count} duplicates ({percentage:.1f})")

# %%
# Check actual height values and their frequencies
print("== Height Value Analysis ==")
print("Most common heights:")
print(df['Height_cm'].value_counts().head(10))

print(f"\nTotal unique height: {df['Height_cm'].nunique()}")
print(f"Height range: {df['Height_cm'].min()} - {df['Height_cm'].max()}")

# %%
# Identify constant features (columns with only one unique value)
# Such features provide no predictive value for supplement category classification
# However, their presence might indicate issues in health data collection
constant_features = [col for col in df.columns if df[col].nunique() == 1]
print("Constant features:", constant_features)

# %%
# Define IQR-based outlier detection function for health metrics
# CHOICE RATIONALE: IQR method chosen for supplement recommendation because:
# 1. Health data often follows non-normal distributions
# 2. Extreme health values are meaningful for supplement targeting (e.g., severe obesity → Detox category)
# 3. Provides interpretable cutoff points for supplement category rules
def find_outliers_IQR_method(input_df, variable):
    IQR = input_df[variable].quantile(0.75) - input_df[variable].quantile(0.25)
    lower_limit = input_df[variable].quantile(0.25) - (IQR*1.5)
    upper_limit = input_df[variable].quantile(0.75) + (IQR*1.5)
    return lower_limit, upper_limit

# Systematic outlier analysis across health metrics relevant for supplement classification
# Focus on key health indicators that will inform supplement category mapping:
# - BMI, Blood Pressure, Glucose → potential Heart Health/Detox candidates
# - Cholesterol → Heart Health supplement candidates
numeriacl_cols = ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'Daily_Caloric_Intake', 'Cholesterol_mg_dL', 
                  'Blood_Pressure_mmHg', 'Clucose_mg_dL', 'Weekly_Exercise_Hours']

for col in numeriacl_cols:
    if col in df.columns:
        lower, upper = find_outliers_IQR_method(df, col)
        outliers_count = len(df[(df[col] < lower) | (df[col] > upper)])
        print(f"{col}: {outliers_count} outliers (range: {lower:.1f} - {upper:.1f})")
        print("*"*50)

# Add to the end of cell 16
# Store BMI outliers for detailed analysis
lower_bmi, upper_bmi = find_outliers_IQR_method(df, 'BMI')
bmi_outliers = df[(df['BMI'] < lower_bmi) | (df['BMI'] > upper_bmi)]
print(f"\n{len(bmi_outliers)} BMI outliers stored successfully")

# %%
# Show outlier details with related columns
print("\nBMI Outlier Details:")
outlier_details = bmi_outliers[['Patient_ID', 'BMI', 'Weight_kg', 'Height_cm', 'Age', 'Disease_Type']].copy()
print(outlier_details.sort_values('BMI'))

# %%
# Validate BMI calculations to ensure accurate supplement category assignment
# Cross-check computed BMI against recorded BMI to prevent misclassification
print("== BMI Calculation Verification ==")
outlier_patients = ['P0456', 'P0324', 'P0839', 'P0831']

for patient_id in outlier_patients:
    patient_data = df[df['Patient_ID'] == patient_id]
    if not patient_data.empty:
        weight = patient_data['Weight_kg'].iloc[0]
        height_cm = patient_data['Height_cm'].iloc[0]
        height_m = height_cm / 100 # Convert cm to meters for BMI formula
        calculated_bmi = weight / (height_m ** 2) # Standard BMI formula: kg/m²
        recorded_bmi = patient_data['BMI'].iloc[0]
        
        print(f"Patient {patient_id}:")
        print(f"  Calculated BMI: {calculated_bmi:.1f}")
        print(f"  Recorded BMI: {recorded_bmi:.1f}")
        print(f"  Match: {abs(calculated_bmi - recorded_bmi) < 0.1}")
        print()

# %%
# DECISION: PRESERVE BMI outliers in the dataset for supplement recommendation model
# SUPPLEMENT BUSINESS JUSTIFICATION: BMI values of 51-52 represent the core target market
# for Korure Detox products. These patients are:
# 1. Most in need of weight management supplement intervention
# 2. Represent high-value customers for supplement business
# 3. Provide crucial training data for Detox category classification
# 4. Mathematically verified as accurate measurements

# %%
#==========================================================================================
# KORURE SUPPLEMENT CATEGORY MAPPING
#==========================================================================================

# Create target variable for supplement recommendation based on health indicators
# This mapping connects individual health conditions to appropriate Korure product categories
# following the rule-based approach outlined in the project proposal

# %% [markdown]
# # Korure Products
# 
# ## Product Categories
# 
# ### Joint Health - MP Oil, MP Powder, Relief Cream
# Target: Age 65+, high BMI + older adults, high exercise hours
# Focus: Joint care and mobility support
# 
# ### Cardiovascular Health - Algae Oil (Plant Omega-3)
# Target: High blood pressure, high cholesterol, diabetes, hypertension
# Focus: Heart health and circulation
# 
# ### Digestive Health - Kiwi Prebiotic + Vitamin C
# Target: Moderate risk group and general health management group
# Focus: Preventive health care and basic nutritional support
# 
# ## Algorithm Priority (Check Order)
# 1. Joint Health (age/mobility issues)
# 2. Cardiovascular Health (heart disease risk)
# 3. Digestive Health (remaining patients)

# %%
# KORURE SUPPLEMENT CATEGORY MAPPING
# 3 Categories: Joint Health, Digestive Health

def assign_supplement_category(row):
    """
    Function to recommend appropriate Korure product category based on patient's health data
    
    Category descriptions:
    - Joint Health: Joint care products (MP Oil, MP Powder, Relief Cream)
    - Cardiovascular Health: Heart health products (Algae Oil - Plant Omega-3)
    - Digestive Health: Digestive support products (Kiwi Prebiotic + Vitamin C)
    """
    
    # Priority 1: Joint Health
    # Age-related joint degeneration, obesity-related joint stress, and overuse from excessive exercise
    if (row['Age'] >= 65 or 
        (row['BMI'] >= 35 and row['Age'] >= 55) or 
        (row['Age'] >= 45 and row['Weekly_Exercise_Hours'] >= 9)):
        return 'Joint Health'
    
    # Priority 2: Cardiovascular Health  
    # Patients with elevated cardiovascular risk markers
    if (row['Blood_Pressure_mmHg'] >= 160 or 
        row['Cholesterol_mg/dL'] >= 240 or
        row['Disease_Type'] in ['Diabetes', 'Hypertension']):
        return 'Cardiovascular Health'
    
    # Priority 3: Digestive Health
    # All remaining patients assigned to digestive health category
    # This includes patients with digestive issues, dietary restrictions, or general health maintenance needs
    return 'Digestive Health'

# %%
# Apply mapping function to create supplement category recommendations
print("Starting Korure product category mapping with 3 categories...")
df['Korure_Category'] = df.apply(assign_supplement_category, axis=1)

# Display mapping results
print("\n== Mapping Results ==")
category_counts = df['Korure_Category'].value_counts()
print(category_counts)

print(f"\nOut of {len(df)} total patients:")
for category, count in category_counts.items():
    percentage = (count/len(df)) * 100
    print(f"- {category}: {count} patients ({percentage:.1f}%)")

# Show sample mapping results
print("\n== Sample Mapping Results ==")
sample_data = df[['Patient_ID', 'Age', 'BMI', 'Disease_Type', 'Physical_Activity_Level', 
                  'Blood_Pressure_mmHg', 'Cholesterol_mg/dL', 'Weekly_Exercise_Hours', 'Korure_Category']].head(10)
print(sample_data.to_string(index=False))

# %% [markdown]
# ## Actual Patient Distribution
# 1. Cardiovascular Health: 47.5%
# 2. Joint Health: 34.6%
# 3. Digestive Health: 17.9%

# %%
# Analyze key characteristics by category
print("\n== Average Health Indicators by Category ==")
category_analysis = df.groupby('Korure_Category').agg({
    'Age': 'mean',
    'BMI': 'mean',
    'Blood_Pressure_mmHg': 'mean',
    'Cholesterol_mg/dL': 'mean',
    'Glucose_mg/dL': 'mean',
    'Weekly_Exercise_Hours': 'mean'
}).round(1)

print(category_analysis.to_string())

print("\n== Final Category Distribution ==")
print("Mapping completed! New 'Korure_Category' column has been created.")
print("Categories mapped to specific Korure products:")
print("- Joint Health → MP Oil, MP Powder, Relief Cream")
print("- Cardiovascular Health → Algae Oil (Plant Omega-3)")
print("- Digestive Health → Kiwi Prebiotic + Vitamin C")
print("\nNote: This 3-category approach ensures balanced sample sizes for machine learning model training.")


