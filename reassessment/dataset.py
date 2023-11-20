import pandas as pd

# Load Datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

paths = [("max_temp", "./datasets/UK Met Office Public Data - Maximum Temperature Series.txt"),
         ("mean_temp", "./datasets/UK Met Office Public Data - Mean Temperature Series.txt"),
         ("min_temp", "./datasets/UK Met Office Public Data - Minimum Temparature Series.txt"),
         ("rainfall", "./datasets/UK Met Office Public Data - mm Rainfall Series.txt"),
         ("sunshine", "./datasets/UK Met Office Public Data - Hours of Strong Sunshine Series.txt"),
         ("rainydays", "./datasets/UK Met Office Public Data - Mean Days 1mm or Above Rainfall Series.txt")
         ]

datasets = []

for name, path in paths:
    # Read data from path
    data = pd.read_csv(path, delim_whitespace=True, skiprows=5)
    # Convert the dataframe into long format
    data = pd.melt(data, id_vars='year', value_vars=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    # Create Date from year value and month
    data['Date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['variable'], format='%Y-%b')
    data = data.drop(columns=['year', 'variable'])
    # Set the Date as index
    data.set_index('Date', inplace=True)
    # Rename the column to an unique name
    data.columns = [name]
    # Sort DataFrame by Date
    data = data.sort_index()
    data.head()
    datasets.append(data)

raw_data = pd.concat(datasets, axis=1, join='inner')

# Clear NaN Values
# Add previous month's rainfall as a new feature
df = raw_data.copy()
df['prev_month'] = df['rainfall'].shift(1)
# Add previous year's rainfall as a new feature
df['prev_year'] = df['rainfall'].shift(12)
# # Add the month of the year
# Extract the month
df['month'] = df.index.month
# Calculate average rainfall for each month
average_rainfall = df.groupby('month')['rainfall'].mean()
df['average_rainfall'] = df['month'].map(average_rainfall)
# Calculate median rainfall for each month
median_rainfall = df.groupby('month')['rainfall'].median()
df['median_rainfall'] = df['month'].map(median_rainfall)

# Function to transform the month to the 0-1 range
# April as 0 and November as 1
# def transform_month(month):
#     # Normalize to 0-1 range with April as 0 and November as 1
#     return (month - 4) / 7 if month >= 4 else (month + 8) / 7
#
# # Apply the transformation
# df['month'] = df['month'].apply(transform_month)
# df = df.drop(['month'], axis=1)


# dataset['month'] = dataset.index.month
df = df.dropna(axis=0)


print(df.describe())
df.to_csv("dataframe.csv")

# Create Train/Test Split here to reduce repetitive code
Y = df['rainfall']
X = df.drop(['rainfall'], axis=1)
print(f'Features: \n {X.columns}')
print()

# Initialize the MinMaxScaler
scaler = MinMaxScaler()
# Fit the scaler to the features and transform them
#X_scaled = scaler.fit_transform(X)

# Separate Train and Test Sets
X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X, Y, test_size=0.30, random_state=0, shuffle=False)
