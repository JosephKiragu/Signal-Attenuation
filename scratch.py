# Convert NaN values to a new category for categorical variables
df['SYNOP Code'] = df['SYNOP Code'].fillna('missing').astype(str)

# Define preprocessor
num_features = ['Absolute Humidity', 'Wind Speed', 'Visibility', 'Time', 
                'Wind Speed Max', 'Temperature', 'Temperature Difference', 
                'Particulate', 'Distance']

cat_features = ['SYNOP Code']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)])

# Append classifier to preprocessing pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestRegressor(n_estimators = 200, random_state = 42, 
                                                           max_depth = None, max_features = 'sqrt',
                                                           min_samples_leaf = 1, min_samples_split = 2))])

# Separate features and target variable
X = df.drop('FSO_Att', axis=1)
y = df['FSO_Att']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

clf.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, r2_score
baseline_predictions = clf.predict(X_test)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
baseline_r2 = r2_score(y_test, baseline_predictions)

feature_scores = {}
for feature in X_test.columns:
    X_test_permuted = X_test.copy()
    X_test_permuted[feature] = np.random.permutation(X_test[feature])
    permuted_predictions = clf.predict(X_test_permuted)
    rmse_permuted = np.sqrt(mean_squared_error(y_test, permuted_predictions))
    r2_permuted = r2_score(y_test, permuted_predictions)
    feature_scores[feature] = {'RMSE_Difference': rmse_permuted - baseline_rmse,
                                'R2_Difference': baseline_r2 - r2_permuted}
    

importance_df = pd.DataFrame(feature_scores).T



importance_df = importance_df.sort_values(by = 'RMSE_Difference', ascending=False)

plt.figure(figsize=(15, 10))
sns.barplot(x='RMSE_Difference', y=importance_df.index, data=importance_df, color='b', label='RMSE Difference')
plt.xlabel('RMSE')
plt.ylabel('Feature')
plt.title('Feature Importance based on RMSE')
plt.show()

plt.figure(figsize=(15, 10))
sns.barplot(x='R2_Difference', y=importance_df.index, data=importance_df, color='g', label='R-squared Difference')
plt.xlabel('R-squared')
plt.ylabel('Feature')
plt.title('Feature Importance based on R-squared')
plt.show()
