from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from dataset import df

def perform_pca(features):
    # Perform PCA
    pca = PCA(n_components=5)
    pca.fit(features)
    # Access the selected features based on explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    selected_features = [feature for feature, ratio in zip(features.columns, explained_variance_ratio) if ratio > 0.01]
    # Print the selected features
    print("Selected Features:")
    for feature in selected_features:
        print(feature)
    return pca


# Get Total Emission
Y = df['Total Emission']
X = df.drop(['Total Emission'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
print(X_pca.shape)
selected_features = [feature for feature, ratio in zip(X.columns, pca.explained_variance_ratio_) if ratio > 0.01]
# Print the selected features
print("Selected Features:")
for feature in selected_features:
    print(feature)

# Separate Train and Test Sets
X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X_pca, Y, test_size=0.30, random_state=0, shuffle=True)

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