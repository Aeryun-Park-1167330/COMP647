# %%
# EXPLORATORY DATA ANALYSIS (EDA) -
# Health Data Analysis for Supplement Recommendation System

import pandas as pd
import matplotlib.pyplot as plt

# Set up matplotlib for better display
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 10

print("== EXPLORATORY DATA ANALYSIS - HEALTH SUPPLEMENT RECOMMENDATION ==")
print("Systematic analysis of patient health data for machine learning model development")

# =============================================================================
# DATA LOADING
# =============================================================================

# Load the preprocessed data
df = pd.read_csv('preprocessed_data.csv')
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# %%
# =============================================================================
# STEP 1: BASIC DATA INFORMATION
# Understanding the dataset structure and composition
# =============================================================================

print("\n" + "="*60)
print("STEP 1: DATASET OVERVIEW")
print("="*60)

# List all variables in the dataset
print("\nDataset variables:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

# Check for missing values
print(f"\nMissing values check:")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("No missing values found - dataset is complete")
else:
    print("Missing values detected:")
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"  - {col}: {missing} missing values")


# %%
# =============================================================================
# STAP 2: TARGET VARIABLE ANALYSIS
# Understanding the distribution of supplement categories
# =============================================================================

# Analyze the distribution of supplement categories
category_counts = df['Korure_Category'].value_counts()
print("Supplement category distribution:")
total_patients = len(df)

for category, count in category_counts.items():
    percentage = (count / total_patients) * 100
    print(f"  - {category}: {count} patients ({percentage:.1f}%)")

# Visualize target variable distribution
plt.figure(figsize=(10, 5))

# Bar chart
plt.subplot(1, 2, 1)
category_counts.plot(kind='bar', color=['lightcoral', 'lightblue', 'lightgreen'])
plt.title('Distribution of Supplement Categories')
plt.xlabel('Category')
plt.ylabel('Number of Patients')
plt.xticks(rotation=45)

# Pie chart
plt.subplot(1, 2, 2)
plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', 
        colors=['lightcoral', 'lightblue', 'lightgreen'])
plt.title('Category Distribution (Percentage)')

plt.tight_layout()
plt.show()


# %%
# =============================================================================
# STEP 3: AGE ANALYSIS
# Examining age patterns and distributions
# =============================================================================

# Age descriptive statistics
age_stats = df['Age'].describe()

# Age distribution visualization
plt.figure(figsize=(8, 5))
plt.hist(df['Age'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
plt.title('Age Distribution of Patients')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.axvline(x=df['Age'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["Age"].mean():.1f} years')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# insights
print(f"Key Insights:")
print(f"   • Average patient age: {df['Age'].mean():.1f} years")
print(f"   • Age range: {df['Age'].min():.0f}-{df['Age'].max():.0f} years")
print(f"   • Most patients are middle-aged to seniors")

# Age distribution
seniors = df[df['Age'] >= 60].shape[0]
print(f"   • Seniors (60+): {seniors} patients ({seniors/len(df)*100:.1f}%) - significant portion")
print(f"   • Distribution appears fairly spread across age groups")

# %%
# =============================================================================
# STEP 4: BMI ANALYSIS
# Body Mass Index distribution and health implications
# =============================================================================

print("\n" + "="*60)
print("STEP 4: BMI (BODY MASS INDEX) ANALYSIS")
print("="*60)

# BMI descriptive statistics
bmi_stats = df['BMI'].describe()
print("BMI descriptive statistics:")
print(f"  - Mean BMI: {bmi_stats['mean']:.1f}")
print(f"  - Median BMI: {bmi_stats['50%']:.1f}")
print(f"  - Standard deviation: {bmi_stats['std']:.1f}")
print(f"  - BMI range: {bmi_stats['min']:.1f} - {bmi_stats['max']:.1f}")

# BMI categories classification
underweight = (df['BMI'] < 18.5).sum()
normal = ((df['BMI'] >= 18.5) & (df['BMI'] < 25)).sum()
overweight = ((df['BMI'] >= 25) & (df['BMI'] < 30)).sum()
obese = (df['BMI'] >= 30).sum()

print(f"\nBMI category distribution:")
print(f"  - Underweight (<18.5): {underweight} patients ({underweight/len(df)*100:.1f}%)")
print(f"  - Normal (18.5-24.9): {normal} patients ({normal/len(df)*100:.1f}%)")
print(f"  - Overweight (25-29.9): {overweight} patients ({overweight/len(df)*100:.1f}%)")
print(f"  - Obese (≥30): {obese} patients ({obese/len(df)*100:.1f}%)")

# BMI distribution visualization
plt.figure(figsize=(8, 5))
plt.hist(df['BMI'], bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
plt.title('BMI Distribution of Patients')
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Frequency')
plt.axvline(x=25, color='orange', linestyle='--', label='Overweight threshold (25)')
plt.axvline(x=30, color='red', linestyle='--', label='Obesity threshold (30)')
plt.axvline(x=df['BMI'].mean(), color='blue', linestyle='-', 
            label=f'Mean BMI: {df["BMI"].mean():.1f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# =============================================================================
# STEP 5: BLOOD PRESSURE ANALYSIS
# Cardiovascular health indicator examination
# =============================================================================

print("\n" + "="*60)
print("STEP 5: BLOOD PRESSURE ANALYSIS")
print("="*60)

# Blood pressure descriptive statistics
bp_stats = df['Blood_Pressure_mmHg'].describe()
print("Blood pressure descriptive statistics:")
print(f"  - Mean BP: {bp_stats['mean']:.1f} mmHg")
print(f"  - Median BP: {bp_stats['50%']:.1f} mmHg")
print(f"  - Standard deviation: {bp_stats['std']:.1f} mmHg")
print(f"  - BP range: {bp_stats['min']:.0f} - {bp_stats['max']:.0f} mmHg")

# Blood pressure categories (based on AHA guidelines)
normal_bp = (df['Blood_Pressure_mmHg'] < 120).sum()
elevated_bp = ((df['Blood_Pressure_mmHg'] >= 120) & (df['Blood_Pressure_mmHg'] < 130)).sum()
high_bp_stage1 = ((df['Blood_Pressure_mmHg'] >= 130) & (df['Blood_Pressure_mmHg'] < 140)).sum()
high_bp_stage2 = (df['Blood_Pressure_mmHg'] >= 140).sum()

print(f"\nBlood pressure category distribution:")
print(f"  - Normal (<120): {normal_bp} patients ({normal_bp/len(df)*100:.1f}%)")
print(f"  - Elevated (120-129): {elevated_bp} patients ({elevated_bp/len(df)*100:.1f}%)")
print(f"  - High Stage 1 (130-139): {high_bp_stage1} patients ({high_bp_stage1/len(df)*100:.1f}%)")
print(f"  - High Stage 2 (≥140): {high_bp_stage2} patients ({high_bp_stage2/len(df)*100:.1f}%)")

# Blood pressure distribution visualization
plt.figure(figsize=(8, 5))
plt.hist(df['Blood_Pressure_mmHg'], bins=20, color='salmon', alpha=0.7, edgecolor='black')
plt.title('Blood Pressure Distribution of Patients')
plt.xlabel('Blood Pressure (mmHg)')
plt.ylabel('Frequency')
plt.axvline(x=120, color='green', linestyle='--', label='Normal threshold (120)')
plt.axvline(x=140, color='red', linestyle='--', label='Hypertension threshold (140)')
plt.axvline(x=df['Blood_Pressure_mmHg'].mean(), color='blue', linestyle='-', 
            label=f'Mean BP: {df["Blood_Pressure_mmHg"].mean():.1f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# =============================================================================
# STEP 6: COMPARATIVE ANALYSIS BY SUPPLEMENT CATEGORY
# Examining health patterns across different supplement categories
# =============================================================================

print("\n" + "="*60)
print("STEP 6: COMPARATIVE ANALYSIS BY SUPPLEMENT CATEGORY")
print("="*60)

# Calculate mean values for key health indicators by category
categories = df['Korure_Category'].unique()

print("Health indicators by supplement category:")
print(f"{'Category':<25} {'Age':<8} {'BMI':<8} {'BP':<8} {'Cholesterol':<12} {'Exercise':<10}")
print("-" * 75)

for category in categories:
    category_data = df[df['Korure_Category'] == category]
    avg_age = category_data['Age'].mean()
    avg_bmi = category_data['BMI'].mean()
    avg_bp = category_data['Blood_Pressure_mmHg'].mean()
    avg_chol = category_data['Cholesterol_mg/dL'].mean()
    avg_exercise = category_data['Weekly_Exercise_Hours'].mean()
    
    print(f"{category:<25} {avg_age:<8.1f} {avg_bmi:<8.1f} {avg_bp:<8.1f} {avg_chol:<12.1f} {avg_exercise:<10.1f}")

# Statistical significance note
print("\nNote: These are descriptive statistics. Statistical significance testing")
print("would be required to confirm meaningful differences between groups.")

# %%
# =============================================================================
# STEP 7: EXERCISE AND HEALTH RELATIONSHIP
# Analyzing the correlation between physical activity and health metrics
# =============================================================================

print("\n" + "="*60)
print("STEP 7: EXERCISE AND HEALTH RELATIONSHIP ANALYSIS")
print("="*60)

# Exercise descriptive statistics
exercise_stats = df['Weekly_Exercise_Hours'].describe()
print("Weekly exercise hours descriptive statistics:")
print(f"  - Mean: {exercise_stats['mean']:.1f} hours/week")
print(f"  - Median: {exercise_stats['50%']:.1f} hours/week")
print(f"  - Standard deviation: {exercise_stats['std']:.1f} hours/week")
print(f"  - Range: {exercise_stats['min']:.1f} - {exercise_stats['max']:.1f} hours/week")

# Exercise vs BMI relationship
plt.figure(figsize=(8, 6))
plt.scatter(df['Weekly_Exercise_Hours'], df['BMI'], alpha=0.6, color='purple')
plt.title('Relationship between Exercise Hours and BMI')
plt.xlabel('Weekly Exercise Hours')
plt.ylabel('Body Mass Index (BMI)')
plt.grid(True, alpha=0.3)

# Add trend line (simple correlation visualization)
correlation = df['Weekly_Exercise_Hours'].corr(df['BMI'])
plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
         transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white'))
plt.show()

print(f"\nCorrelation between exercise and BMI: {correlation:.3f}")
if abs(correlation) > 0.3:
    print("Moderate correlation detected")
elif abs(correlation) > 0.1:
    print("Weak correlation detected")
else:
    print("No significant correlation detected")

# %%
# =============================================================================
# STEP 8: Physical activity ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("STEP 8: Physical activity ANALYSIS")
print("="*60)

# Physical activity level distribution
plt.figure(figsize=(8, 5))
activity_counts = df['Physical_Activity_Level'].value_counts()
plt.pie(activity_counts.values, labels=activity_counts.index, autopct='%1.1f%%')
plt.title('Physical Activity Level Distribution')
plt.show()

# Physical activity analysis
print("\nPhysical activity level distribution:")
for activity, count in activity_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  - {activity}: {count} patients ({percentage:.1f}%)")

print("\nKey insight: Perfect balance across activity levels - ideal for analysis")

# Activity level vs supplement preference analysis
print("\nActivity vs Supplement preference:")
activity_supplement_pct = pd.crosstab(df['Physical_Activity_Level'], df['Korure_Category'], normalize='index') * 100
print(activity_supplement_pct.round(1))

# %% [markdown]
# # SUMMARY AND KEY INSIGHTS
# 
# ## DATASET SUMMARY:
#   - Total patients analyzed: 1000
#   - Supplement categories: 3
#   - Complete data (no missing values): No
# 
# ## PATIENT DEMOGRAPHICS:
#   - Average age: 49.9 years
#   - Average BMI: 28.2
#   - Hypertensive patients (BP ≥140): 572 (57.2%)
#   - Obese patients (BMI ≥30): 393 (39.3%)
# 
# ## DATA QUALITY ASSESSMENT:
#   - Class balance: Imbalanced
#   - Data completeness: 100%
#   - Suitable for machine learning: Yes
# 
# ## RECOMMendations FOR MODEL DEVELOPMENT:
#   1. Data is ready for machine learning algorithms
#   2. Consider feature scaling for algorithms sensitive to magnitude
#   3. Monitor for overfitting due to multiple health indicators
#   4. Cross-validation recommended for model evaluation


