import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the dataset
print("Loading dataset...")
df = sns.load_dataset('titanic')

# =====================================================================
# 1. Ask meaningful questions about the dataset before analysis.
# =====================================================================
print("\n--- 1. Meaningful Questions ---")
print("Q1: What is the overall survival rate of passengers?")
print("Q2: Does socio-economic status (passenger class) impact survival?")
print("Q3: Is there a significant relationship between gender and survival?")

# =====================================================================
# 2. Explore the data structure, including variables and data types.
# =====================================================================
print("\n--- 2. Data Structure & Variables ---")
print("\nFirst 5 rows of the dataset:")
print(df.head()) # Use print(df.head()) if not in Jupyter

print("\nData Types and Non-Null Counts:")
df.info()

# =====================================================================
# 3. Detect potential data issues or problems.
# =====================================================================
print("\n--- 3. Detecting Data Issues (Missing Values) ---")
missing_values = df.isnull().sum()
missing_percentages = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({'Missing Count': missing_values, 'Percentage (%)': missing_percentages})
print(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percentage (%)', ascending=False))

print("\nAction Plan for Issues:")
print("- 'deck': Missing ~77% of data. We will drop this column for analysis.")
print("- 'age': Missing ~20%. We will impute these with the median age.")
print("- 'embarked'/'embark_town': Missing <1%. We will drop these specific rows.")

# Clean the issues detected
df = df.drop(columns=['deck'])
df['age'] = df['age'].fillna(df['age'].median())
df = df.dropna(subset=['embarked', 'embark_town'])


# =====================================================================
# 4. Identify trends, patterns, and anomalies within the data.
# =====================================================================
print("\n--- 4. Trends, Patterns & Anomalies ---")
print("\nStatistical Summary of Numerical Variables:")
print(df.describe())

# Identifying an anomaly (Outliers in Fare)
print("\nAnomaly Detection:")
max_fare = df['fare'].max()
print(f"The maximum fare paid was ${max_fare:.2f}, which is significantly higher than the 75th percentile (${df['fare'].quantile(0.75):.2f}). This is an outlier.")


# =====================================================================
# 5. Test hypotheses and validate assumptions using statistics & visualization.
# =====================================================================
print("\n--- 5. Hypothesis Testing ---")

# Hypothesis: Gender significantly affects survival rates.
# Assumption: Women were more likely to survive than men.
print("\nTesting Hypothesis: Gender vs. Survival")

# Statistical Test (Chi-Square)
contingency_table = pd.crosstab(df['sex'], df['survived'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square P-Value: {p_value}")
if p_value < 0.05:
    print("Conclusion: Reject the null hypothesis. There IS a statistically significant relationship between gender and survival.")
else:
    print("Conclusion: Fail to reject the null hypothesis. No significant relationship.")

# Visualization to validate the assumption
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='sex', y='survived', errorbar=None, palette='pastel')
plt.title('Survival Rate by Gender (Hypothesis Validation)', fontweight='bold')
plt.ylabel('Survival Probability')
plt.xlabel('Gender')
plt.show()

# Insight derived:
print("\nValidation: The visualization confirms that females had a roughly 74% survival rate, while males had less than 20%.")