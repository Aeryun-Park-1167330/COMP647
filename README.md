# Supplement Recommendation System
*COMP647 – Machine Learning Project*
Machine Learning Classification Model for Health-Based Product Suggestions

## Project Overview
This project develops a machine learning classification system that recommends supplement categories based on health and lifestyle data. Using a dataset of 1,000 individuals with comprehensive health metrics, the model predicts suitable supplement categories: Cardiovascular Health, Joint Health, and Digestive Health.

Note: This is an educational project for machine learning practice. The supplement categories are fictional mappings and not actual medical recommendations.

## Dataset Analysis
**Data Source**: Diet Recommendations Dataset (Kaggle)  
**Sample Size**: 1,000 individuals  
**Features**: 21 health and lifestyle variables including age, BMI, disease type, physical activity, cholesterol, blood pressure, and more.

## Tech Stack
- **Python Libraries**: Pandas, NumPy, Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Explainable AI**: SHAP (SHapley Additive exPlanations)
- **Development**: Jupyter Notebook

## Project Assignments

### Assignment 2: Data Preprocessing & EDA
- Handled missing values and outliers
- Exploratory data analysis with visualizations
- Data cleaning and preparation

### Assignment 3: Model Development & Evaluation
- **Feature Engineering**: Created domain-specific features (BMI categories, health risk scores, age groups)
- **Feature Selection**: Used ensemble voting (F-test, Mutual Information, Random Forest)
- **Model Training**: Tested 4 algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- **Hyperparameter Tuning**: GridSearchCV for optimization
- **Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1-score, confusion matrix)
- **Explainable AI**: SHAP values, feature importance, partial dependence plots

**Best Model**: Random Forest with 95%+ accuracy across all classes

## Inspiration
This project was inspired by Korure, a New Zealand-based health supplement brand, to explore how AI can enhance personalized wellness recommendations in the supplement industry.

## Disclaimer
This project is for educational purposes only and does not provide real medical recommendations. The supplement recommendations are based on fictional mappings and should not be considered medical advice. Always consult healthcare professionals for supplement and health decisions.

## Project Status
**Completed**: Assignment 3 (Model Development & Explainable AI)  
*Final project showcasing the complete ML pipeline from data preprocessing to model interpretation*

## Key Learnings
- Feature engineering can be more impactful than algorithm selection
- Multiple evaluation metrics are essential beyond just accuracy
- Explainability is crucial for healthcare applications
- Cross-validation helps prevent overfitting

## References
**Korure**: https://korure.co.nz/ - New Zealand health supplement brand
**Dataset**: www.kaggle.com/datasets/ziya07/diet-recommendations-dataset/data
