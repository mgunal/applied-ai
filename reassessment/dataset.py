import pandas as pd

# Load Datasets
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

dataset = pd.concat(datasets, axis=1, join='inner')
# Add previous month's rainfall as a new feature
dataset['prev_month'] = dataset['rainfall'].shift(1)
dataset['prev_year'] = dataset['rainfall'].shift(12)

# Clear NaN Values
df = dataset.dropna(axis=0)

print(df.describe())
df.to_csv("dataframe.csv")
