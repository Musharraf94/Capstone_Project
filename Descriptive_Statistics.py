import pandas as pd

# File path
file_path = 'C:/Users/mmira/Desktop/ARIZONA/Capstone_Project/data.csv'

# Load data
data = pd.read_csv(file_path)

# Continuous variables
continuous_vars = ['month', 'wght', 'numchldrn', 'agem']
continuous_stats = data[continuous_vars].describe().transpose()
continuous_stats['count'] = data[continuous_vars].count()

# Categorical variables
categorical_vars = ['stunting', 'brthord', 'birthint', 'gender', 'educlvl', 'wlth', 'area']
categorical_stats = pd.DataFrame()

for col in categorical_vars:
    temp = data[col].value_counts().rename_axis('unique_values').reset_index(name='counts')
    temp['percentage'] = (temp['counts'] / temp['counts'].sum()) * 100
    temp['variable'] = col
    categorical_stats = pd.concat([categorical_stats, temp[['variable', 'unique_values', 'percentage', 'counts']]], ignore_index=True)

# Combine continuous and categorical stats for output
final_stats = pd.concat([continuous_stats, categorical_stats.set_index('variable')])
final_stats.rename(columns={'mean': 'Mean', '50%': 'Median', 'std': 'Standard Deviation', 'min': 'Minimum', 
                            'max': 'Maximum', 'count': 'Number of Observations', 'percentage': '%'}, inplace=True)

# Save the stats to a CSV file
final_stats.to_csv('C:/Users/mmira/Desktop/ARIZONA/Capstone_Project/desc_stats.csv')

# Print the stats
print(final_stats)