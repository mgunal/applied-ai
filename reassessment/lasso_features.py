from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from dataset import df
print(df.head())
print()


Y = df['rainfall']
X = df.drop(['rainfall'], axis=1)
print(f'Features: \n {X.columns}')
print()

# Scale your features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(cv=5, random_state=0)
lasso.fit(X_scaled, Y)

# Get the selected features (those with non-zero coefficients)
selected_features = X.columns[lasso.coef_ != 0]
print(f"Lasso Selected Features: \n {selected_features}")
X_lasso = df[selected_features]
