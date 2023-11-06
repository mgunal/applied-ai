from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Import prepared data from datasets
from sklearn.neural_network import MLPRegressor

from dataset import df
print(df.head())
print()


Y = df['rainfall']
X = df.drop(['rainfall'], axis=1)
print(f'Features: {X.columns}')
print()

# Separate Train and Test Sets
X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X, Y, test_size=0.30, random_state=0, shuffle=False)

models = []
# Linear Regression model
linear_model = LinearRegression()
models.append(('Linear Regression', linear_model))

# Artificial Neural Networks (Multi-layer Perceptron) model
ann_model = MLPRegressor(hidden_layer_sizes=(20, 40, 40, 20), activation='relu', solver='adam', random_state=1, learning_rate='adaptive')
models.append(('Artificial Neural Network', ann_model))

# Gradient Boosting
gb_model = GradientBoostingRegressor()
models.append(('Gradient Boosting', gb_model))

if __name__ == '__main__':
    # Run Models with All features
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
