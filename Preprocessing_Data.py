import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load the data
file_path = 'C:/Users/mmira/Desktop/ARIZONA/Capstone_Project/data.csv'
data = pd.read_csv(file_path)

# Preprocessing Pipeline
numeric_cols = ['month', 'wght', 'agem', 'numchldrn']
categorical_cols = ['gender', 'brthord', 'birthint', 'educlvl', 'wlth', 'area']

# Numeric features scaling
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Categorical features encoding
encoder = OneHotEncoder(drop='first')  # Avoid multicollinearity
encoded_features = encoder.fit_transform(data[categorical_cols])
encoded_features_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out())

# Combine numeric and encoded categorical features
data.drop(categorical_cols, axis=1, inplace=True)
data = pd.concat([data, encoded_features_df], axis=1)

# Outlier detection and removal
Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[~((data[numeric_cols] < lower_bound) | (data[numeric_cols] > upper_bound)).any(axis=1)]

# Print summary statistics and data types
print(data.describe())
print(data.dtypes)
print(data.head())

# Correlation matrix
corr_matrix = data.corr()
print(corr_matrix)

# Save the processed data
data.to_csv('C:/Users/mmira/Desktop/ARIZONA/Capstone_Project/processed_data.csv', index=False)
