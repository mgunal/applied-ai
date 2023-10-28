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
Y = df['Total Emission']
X = df[['Total Energy', 'Fraction Energy Renewable']]
#X = df[['Total Energy Renewable', 'Total Energy Fuels', 'Coal Emission', 'Oil Emission', 'Gas Emission']]
print(f'Features: {X.columns}')
# Scale your features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X

# Separate Train and Test Sets
X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X_scaled, Y, test_size=0.30, random_state=0, shuffle=True)

models = []
# Linear Regression model
linear_model = LinearRegression()
#models.append(('Linear Regression', linear_model))

# Artificial Neural Networks (Multi-layer Perceptron) model
ann_model = MLPRegressor(hidden_layer_sizes=(100), activation='relu', solver='adam', random_state=1,
                         learning_rate='adaptive')
#models.append(('Artificial Neural Network', ann_model))

# Gradient Boosting
gb_model = GradientBoostingRegressor(
        learning_rate=0.1,
        n_estimators=52,
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=0
        )
models.append(('Gradient Boosting', gb_model))

# Support Vector Machine (SVR) model
svm_model = SVR()
#models.append(('Support Vector Machine', svm_model))

# Decision Trees
dt_model = DecisionTreeRegressor()
#models.append(('Decision Tree', dt_model))

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
