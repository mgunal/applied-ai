import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#
# CO2 Dataset
co2_df = pd.read_csv("dataset/GCB2022v27_MtCO2_flat.csv")
print(f"CO2 Dataset: \n {co2_df.columns}")
# Extract Needed Data
ukco2_df = co2_df[(co2_df['Year'] >= 1990) & (co2_df['Year'] <= 2020) & (co2_df['Country'] == 'United Kingdom')]
ukco2_df.set_index(ukco2_df['Year'], inplace=True)
# Drop Unneeded columns
ukco2_df = ukco2_df.drop(["Country", "ISO 3166-1 alpha-3", "Year"], axis=1)

columns = {'Total': 'Total Emission',
           'Coal': 'Coal Emission',
           'Oil': 'Oil Emission',
           'Gas': 'Gas Emission',
           'Cement': 'Cement Emission',
           'Flaring': 'Flaring Emission',
           'Other': 'Other Emission'}
ukco2_df.rename(columns=columns, inplace=True)


#
# UK Renewable Energy Dataset
ukrn_df = pd.read_csv("dataset/uk_renewable_energy.csv")
print(f"UK Renewable Energy Dataset: \n {ukrn_df.columns}")
# Rename columns using a dictionary
columns = {'Energy from renewable & waste sources': 'Total Energy Renewable',
           'Total energy consumption of primary fuels and equivalents': 'Total Energy Fuels',
           'Fraction from renewable sources and waste': 'Fraction Energy Renewable',
           'Hydroelectric power': 'Hydro Energy',
           'Wind, wave, tidal': 'Wind Energy',
           'Solar photovoltaic': 'Solar Energy',
           'Geothermal aquifers': 'Geo Energy',
           'Landfill gas': 'Landfill Energy',
           'Sewage gas': 'Sewage Energy',
           'Biogas from autogen': 'Biogas Energy',
           'Municipal solid waste (MSW)': 'Solid Waste Energy'}

ukrn_df.rename(columns=columns, inplace=True)
ukrn_df.set_index(ukrn_df['Year'], inplace=True)
# Drop unneeded columns
ukrn_df = ukrn_df.drop(['Year'], axis=1)

# Merge Datasets
df = pd.concat([ukrn_df, ukco2_df], axis=1)

# Create Total Energy
df['Total Energy'] = df['Total Energy Renewable'] + df['Total Energy Fuels']

# Per Capita is highly correlated with the Total Emission and that can't be provided as a feature
df = df.drop('Per Capita', axis=1)
print(f"Final Dataset \n  {df.columns}")

correlation = df.corr()['Total Emission']
print("Correlation: ")
print(correlation)
correlation.to_csv('./correlation.csv')