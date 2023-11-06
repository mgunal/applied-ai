from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import df
print(df.head())
print()


Y = df['rainfall']
X = df.drop(['rainfall'], axis=1)
print(f'Features: \n {X.columns}')
print()

# Separate Train and Test Sets
X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X, Y, test_size=0.30, random_state=0, shuffle=False)

# # Scale your features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

lasso_cv = LassoCV(cv=5, random_state=0)
# Fit the LassoCV model to the training data
lasso_cv.fit(X_train, Y_train)

# Determine the best alpha parameter found by cross-validation
best_alpha = lasso_cv.alpha_

# Fit a Lasso regression with the best alpha to the training data
lasso = Lasso(alpha=best_alpha)
lasso.fit(X_train, Y_train)

# Identify the features that Lasso selected (non-zero coefficients)
lasso_coefficients = lasso.coef_
selected_features = X_train.columns[lasso_coefficients != 0]

print(f"Lasso Selected Features: \n {selected_features}")
X_lasso = df[selected_features]
