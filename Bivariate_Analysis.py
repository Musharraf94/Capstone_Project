import pandas as pd

# File path
file_path = 'C:/Users/mmira/Desktop/ARIZONA/Capstone_Project/data.csv'

# Load data
data = pd.read_csv(file_path)

# Define the categorical variables
categorical_vars = ['brthord', 'birthint', 'gender', 'educlvl', 'wlth', 'area']

# Create a DataFrame to store results
final_results = pd.DataFrame()

# Calculate the percentage of children stunting ('stunting') for each category
for var in categorical_vars:
    # Create a crosstabulation
    ct = pd.crosstab(data[var], data['stunting'], normalize='index') * 100
    # Rename columns for clarity
    ct.columns = ['No Stunting (%)', 'Stunting (%)']
    # Calculate the number of observations per category
    ct['Number of Observations'] = data[var].value_counts()
    # Add a multi-level index for clarity
    ct.index = pd.MultiIndex.from_product([[var], ct.index], names=['Variable', 'Category'])
    # Append the result to the final DataFrame
    final_results = pd.concat([final_results, ct])

# Reset index for a cleaner look
final_results.reset_index(inplace=True)

# Save the stats to a CSV file
final_results.to_csv('C:/Users/mmira/Desktop/ARIZONA/Capstone_Project/bivariate_analysis.csv')

# Print the stats
print(final_results)