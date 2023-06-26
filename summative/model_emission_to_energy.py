from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from dataset import df
import pandas as pd

# Get Total Emission
print(df.columns)

# CO2 Emissions from the List of energy sources
sources = ['Coal Emission', 'Oil Emission', 'Gas Emission'] #, 'Cement Emission', 'Flaring Emission', 'Other Emission']
X = df[sources]
print(f'Features: {X.columns}')
# Total Energy generated from the Fuels
Y = df['Total Energy Fuels']

# Scale your features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)


# Separate Train and Test Sets
X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X, Y, test_size=0.30, random_state=0, shuffle=True)

models = []
# Linear Regression model
linear_model = LinearRegression()
models.append(('Linear Regression', linear_model))

# Artificial Neural Networks (Multi-layer Perceptron) model
ann_model = MLPRegressor(hidden_layer_sizes=(100), activation='relu', solver='adam', random_state=1,
                         learning_rate='adaptive')
models.append(('Artificial Neural Network', ann_model))

# Gradient Boosting
gb_model = GradientBoostingRegressor()
models.append(('Gradient Boosting', gb_model))

# Support Vector Machine (SVR) model
svm_model = SVR()
models.append(('Support Vector Machine', svm_model))

# Decision Trees
dt_model = DecisionTreeRegressor()
models.append(('Decision Tree', dt_model))

for model_name, model in models:
    print(f"{model_name} Model")
    model.fit(X_train, Y_train)

    prediction = model.predict(X_validation)
    mse = mean_squared_error(Y_validation, prediction)
    mae = mean_absolute_error(Y_validation, prediction)
    r2 = r2_score(Y_validation, prediction)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R2 Score:", r2)
    print()

# Assuming 'model' is your trained Linear Regression model
print("Linear Model Parameters:")
coefficients = linear_model.coef_
intercept = linear_model.intercept_

for feature, coeff in zip(sources, coefficients):
    print(f"The coefficient for {feature} is {coeff}")
print(f"The intercept is {intercept}")
print()

# Assuming 'gb_model' is your trained Gradient Boosting model
print("Gradient Boosting Model Parameters:")
importances = gb_model.feature_importances_

for feature, importance in zip(sources, importances):
    print(f"The importance of {feature} is {importance}")
