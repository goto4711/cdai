from learntools.core import *
import pandas as pd
import numpy as np
import sys

class Exercise0(CodingProblem):
    _vars = ['congress_114']
    _hint = ("Use the .head() method to display the first few rows of a DataFrame. "
             "This helps you understand the structure and content of your data. "
             "Try: dataframe_name.head(n) where n is the number of rows to display.")
    _solution = CS("congress_114.head(5)")
    
    def check(self, congress_114):
        if not isinstance(congress_114, pd.DataFrame):
            raise AssertionError("Make sure 'congress_114' is a pandas DataFrame.")
        
        if congress_114.empty:
            raise AssertionError("The DataFrame appears to be empty. Make sure you've loaded the data correctly.")
        
        # Check that it has reasonable structure for congress data
        if len(congress_114.columns) < 5:
            raise AssertionError("The DataFrame seems to have too few columns. Make sure you've loaded the complete dataset.")


class Exercise1(CodingProblem):
    _vars = ['congress_114']
    _hint = ("Use the .tail() method to display the last few rows of a DataFrame. "
             "This is useful for checking the end of your dataset and ensuring data loaded completely. "
             "The syntax is similar to .head(): dataframe_name.tail(n)")
    _solution = CS("congress_114.tail(5)")
    
    def check(self, congress_114):
        if not isinstance(congress_114, pd.DataFrame):
            raise AssertionError("Make sure 'congress_114' is a pandas DataFrame.")
        
        if congress_114.empty:
            raise AssertionError("The DataFrame appears to be empty. Make sure you've loaded the data correctly.")


class Exercise2(CodingProblem):
    _vars = ['congress_114']
    _hint = ("Use the .dropna() method to remove rows containing missing values (NaN). "
             "This is important for cleaning your data before analysis. "
             "Remember to assign the result back: df = df.dropna()")
    _solution = CS("congress_114 = congress_114.dropna()")
    
    def check(self, congress_114):
        nan_count = congress_114.isnull().sum().sum()
        if nan_count > 0:
            raise AssertionError(f"Found {nan_count} NaN values remaining in the DataFrame. "
                               "Make sure you used dropna() and assigned the result back to congress_114.")
        
        if congress_114.empty:
            raise AssertionError("The DataFrame is empty after dropping NaN values. "
                               "This might indicate all rows had missing values.")


class Exercise3(ThoughtExperiment):
    _hint = ("Create a subset of the DataFrame containing only the voting columns (bill_cols). "
             "Use .loc with column indexing: df.loc[:, column_list] selects all rows and specified columns. "
             "Then use .head() to display the first few rows of this subset.")
    _solution = CS("""congress_114_voting = congress_114.loc[:, bill_cols]
congress_114_voting.head()""")


class Exercise4(CodingProblem):
    _vars = ['congress_114_result', 'kmeans_5']
    _hint = ("Add the cluster labels from your KMeans model as a new column to the DataFrame. "
             "KMeans labels are accessed via .labels_ attribute. Convert to numpy array for consistency: "
             "df['new_column'] = np.array(model.labels_)")
    _solution = CS("congress_114_result['cluster_5'] = np.array(kmeans_5.labels_)")
    
    def check(self, congress_114_result, kmeans_5):
        if 'cluster_5' not in congress_114_result.columns:
            raise AssertionError("Column 'cluster_5' not found in congress_114_result. "
                               "Make sure you added the cluster labels as a new column.")
        
        expected_labels = np.array(kmeans_5.labels_)
        actual_labels = congress_114_result["cluster_5"].values
        
        if not np.array_equal(actual_labels, expected_labels):
            raise AssertionError("The 'cluster_5' column doesn't match the KMeans labels. "
                               "Make sure you used np.array(kmeans_5.labels_).")
        
        # Check reasonable number of clusters
        unique_clusters = len(np.unique(actual_labels))
        if unique_clusters > 10:
            raise AssertionError(f"Found {unique_clusters} unique clusters, which seems too many. "
                               "Check your KMeans configuration.")


class Exercise5(ThoughtExperiment):
    _hint = ("Use pd.crosstab() to create a cross-tabulation showing the relationship between party affiliation and cluster assignments. "
             "This helps analyze whether the clustering captured political party differences. "
             "Syntax: pd.crosstab(row_variable, column_variable)")
    _solution = CS("""pd.crosstab(congress_114_result.party, congress_114_result.cluster_5)""")


class Exercise6(ThoughtExperiment):
    _hint = ("Import MaxAbsScaler from sklearn.preprocessing to normalize your data. "
             "MaxAbsScaler scales features to [-1, 1] range by dividing by the maximum absolute value. "
             "Steps: 1) Create scaler, 2) Copy DataFrame, 3) Transform all columns except the last (target), 4) Display result.")
    _solution = CS("""from sklearn.preprocessing import MaxAbsScaler
scaler2 = MaxAbsScaler()
music_normalized_df2 = music_df.copy()
music_normalized_df2.iloc[:,:-1] = scaler2.fit_transform(music_normalized_df2.iloc[:,:-1])
music_normalized_df2.head()""")


class Exercise7(CodingProblem):
    _vars = ['train_set', 'test_set']
    _hint = ("Use train_test_split from sklearn.model_selection to split your data into training and testing sets. "
             "Set test_size=0.2 for 80/20 split and random_state=42 for reproducible results. "
             "Syntax: train, test = train_test_split(data, test_size=0.2, random_state=42)")
    _solution = CS("train_set, test_set = train_test_split(music_df, test_size=0.2, random_state=42)")
    
    def check(self, train_set, test_set):
        if not isinstance(train_set, pd.DataFrame):
            raise AssertionError("train_set should be a pandas DataFrame.")
        
        if not isinstance(test_set, pd.DataFrame):
            raise AssertionError("test_set should be a pandas DataFrame.")
        
        total_expected = 114000
        total_actual = len(train_set) + len(test_set)
        if total_actual != total_expected:
            raise AssertionError(f"Expected total of {total_expected} samples, but got {total_actual}. "
                               "Check your data loading.")
        
        test_ratio = len(test_set) / total_actual
        if abs(test_ratio - 0.2) > 0.01:
            raise AssertionError(f"Test set should be ~20% of data, but got {test_ratio:.1%}. "
                               "Make sure test_size=0.2.")


class Exercise8(ThoughtExperiment):
    _hint = ("Calculate the mean (average) of the 'valence' feature in your training set. "
             "Use np.mean() function: np.mean(dataframe['column_name']). "
             "This gives you a baseline value for the valence feature.")
    _solution = CS("np.mean(train_set['valence'])")


class Exercise9(ThoughtExperiment):
    _hint = ("Import and use mean_absolute_error from sklearn.metrics to evaluate model performance. "
             "MAE measures average absolute difference between predictions and actual values. "
             "Syntax: mean_absolute_error(actual_values, predicted_values)")
    _solution = CS("""from sklearn.metrics import mean_absolute_error
mean_absolute_error(test_set['dance_label'], preds)""")


class Exercise10(ThoughtExperiment):
    _hint = ("Create a simple baseline model using the 'liveness' feature. "
             "Steps: 1) Calculate mean liveness from training set, 2) Predict 1 if test liveness < mean, else 0, "
             "3) Convert boolean to int with .astype(int), 4) Calculate MAE to evaluate performance.")
    _solution = CS("""mean_liveness = np.mean(train_set['liveness'])
preds = (test_set['liveness'].values < mean_liveness).astype(int)
print('Preds: ', preds)
print('Mean absolute error: ', mean_absolute_error(test_set['dance_label'], preds))""")


class Exercise11(ThoughtExperiment):
    _hint = ("Test the 'acousticness' feature using your custom score_ function. "
             "Calculate the mean acousticness from training data, then use score_() to evaluate "
             "how well this threshold separates the dance_label classes.")
    _solution = CS("""mean_acousticness = np.mean(train_set['acousticness'])
print(score_(test_set['acousticness'].values, test_set['dance_label'].values, mean_acousticness))""")


class Exercise12(ThoughtExperiment):
    _hint = ("Define a function to find the optimal threshold for any feature column. "
             "The function should: 1) Sample data for efficiency, 2) Test all unique values as thresholds, "
             "3) Calculate score for each, 4) Return the feature name, best threshold, and best score.")
    _solution = CS("""def min_column(df, col_):
    df = df.sample(frac=0.1, random_state=4711) 
    unq = df[col_].dropna().unique()
    s = np.array([score_(df[col_], df['dance_label'], o) for o in unq if not np.isnan(o)])
    idx = s.argmin()
    return (col_, float(unq[idx]), float(s[idx]))

min_column(train_set, 'valence')""")


class Exercise13(ThoughtExperiment):
    _hint = ("Apply the min_column function to all feature columns (excluding the target 'dance_label'). "
             "Use a for loop to iterate through train_set.columns[:-1] (all columns except the last). "
             "This will show you the best threshold for each feature.")
    _solution = CS("""for w in list(train_set.columns[:-1]):
    print(min_column(train_set, w))""")


class Exercise14(ThoughtExperiment):
    _hint = ("After splitting on 'valence', find the best features for the right split subset. "
             "First, get column names excluding 'valence' and 'dance_label', then apply min_column to each. "
             "This implements a simple decision tree approach by finding the next best split.")
    _solution = CS("""right_split_valence_columns = [c for c in right_split_valence.columns if c not in ['valence', 'dance_label']]
for w in right_split_valence_columns:
    print(min_column(right_split_valence, w))""")


class Exercise15(CodingProblem):
    _vars = ['X_train', 'y_train', 'X_test', 'y_test']
    _hint = ("Prepare data for scikit-learn models by separating features (X) from target (y). "
             "Use .loc to select all columns except 'dance_label' for features, and .values to convert to numpy arrays. "
             "X should contain all features, y should contain only the target variable.")
    _solution = CS("""X_train = train_set.loc[:, train_set.columns != 'dance_label'].values
y_train = train_set['dance_label'].values
X_test = test_set.loc[:, test_set.columns != 'dance_label'].values
y_test = test_set['dance_label'].values""")
    
    def check(self, X_train, y_train, X_test, y_test):
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise AssertionError("X_train and y_train should be numpy arrays. Use .values to convert from pandas.")
        
        if not isinstance(X_test, np.ndarray) or not isinstance(y_test, np.ndarray):
            raise AssertionError("X_test and y_test should be numpy arrays. Use .values to convert from pandas.")
        
        if X_train.shape[0] != y_train.shape[0]:
            raise AssertionError(f"X_train has {X_train.shape[0]} samples but y_train has {y_train.shape[0]}. "
                               "They should match.")
        
        if X_test.shape[0] != y_test.shape[0]:
            raise AssertionError(f"X_test has {X_test.shape[0]} samples but y_test has {y_test.shape[0]}. "
                               "They should match.")
        
        if X_train.shape[1] != 9:
            raise AssertionError(f"Expected 9 features in X_train, but got {X_train.shape[1]}. "
                               "Make sure you excluded only the 'dance_label' column.")


class Exercise16(CodingProblem):
    _vars = ['model']
    _hint = ("Import DecisionTreeClassifier from sklearn.tree and create a model with specific parameters. "
             "Use max_leaf_nodes=4 to limit tree complexity and random_state=4711 for reproducibility. "
             "Then fit the model using .fit(X_train, y_train).")
    _solution = CS("""from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_leaf_nodes=4, random_state=4711).fit(X_train, y_train)""")
    
    def check(self, model):
        from sklearn.tree import DecisionTreeClassifier
        
        if not isinstance(model, DecisionTreeClassifier):
            raise AssertionError("model should be an instance of DecisionTreeClassifier.")
        
        if model.max_leaf_nodes != 4:
            raise AssertionError(f"max_leaf_nodes should be 4, but got {model.max_leaf_nodes}.")
        
        if model.random_state != 4711:
            raise AssertionError(f"random_state should be 4711, but got {model.random_state}.")
        
        if not hasattr(model, 'tree_'):
            raise AssertionError("The model should be fitted. Use .fit(X_train, y_train).")


class Exercise17(CodingProblem):
    _vars = ['preds']
    _hint = ("Use the trained model to make predictions on the test set with .predict(). "
             "Then calculate the mean absolute error to evaluate performance. "
             "This measures how often the model's predictions differ from actual labels.")
    _solution = CS("""preds = model.predict(X_test)
mean_absolute_error(y_test, preds)""")
    
    def check(self, preds):
        if not isinstance(preds, np.ndarray):
            raise AssertionError("Predictions should be a numpy array. Use model.predict(X_test).")
        
        expected_length = 22800
        if len(preds) != expected_length:
            raise AssertionError(f"Expected {expected_length} predictions, but got {len(preds)}. "
                               "Make sure you're predicting on the full test set.")


class Exercise18(CodingProblem):
    _vars = ['preds']
    _hint = ("Use model_2 (the second decision tree) to make predictions on the test set. "
             "Compare its performance with the first model using mean_absolute_error. "
             "Different model parameters can lead to different performance.")
    _solution = CS("""preds = model_2.predict(X_test)
mean_absolute_error(y_test, preds)""")
    
    def check(self, preds):
        if not isinstance(preds, np.ndarray):
            raise AssertionError("Predictions should be a numpy array. Use model_2.predict(X_test).")
        
        expected_length = 22800
        if len(preds) != expected_length:
            raise AssertionError(f"Expected {expected_length} predictions, but got {len(preds)}. "
                               "Make sure you're predicting on the full test set.")


class Exercise19(CodingProblem):
    _vars = ['rf']
    _hint = ("Create a Random Forest classifier with specific parameters and train it. "
             "Use n_estimators=10 (number of trees), min_samples_leaf=5 (minimum samples per leaf), "
             "and random_state=4711. Then fit it and calculate MAE on test predictions.")
    _solution = CS("""from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(10, min_samples_leaf=5, random_state=4711)
rf.fit(X_train, y_train)
mean_absolute_error(y_test, rf.predict(X_test))""")
    
    def check(self, rf):
        from sklearn.ensemble import RandomForestClassifier
        
        if not isinstance(rf, RandomForestClassifier):
            raise AssertionError("rf should be an instance of RandomForestClassifier.")
        
        if rf.n_estimators != 10:
            raise AssertionError(f"n_estimators should be 10, but got {rf.n_estimators}.")
        
        if rf.min_samples_leaf != 5:
            raise AssertionError(f"min_samples_leaf should be 5, but got {rf.min_samples_leaf}.")
        
        if not hasattr(rf, 'estimators_'):
            raise AssertionError("The Random Forest should be fitted. Use .fit(X_train, y_train).")


class Exercise20(ThoughtExperiment):
    _hint = ("Visualize feature importance from the Random Forest model to understand which features are most predictive. "
             "Create a pandas Series from rf.feature_importances_ with column names as index, "
             "sort by importance, and create a bar plot. This shows which audio features matter most for predicting danceability.")
    _solution = CS("""import pandas as pd
forest_importances = pd.Series(rf.feature_importances_, index=train_set.columns[:-1])
fig, ax = plt.subplots()
forest_importances.sort_values(ascending=False).plot.bar(ax=ax) 
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()""")


qvars = bind_exercises(globals(), [
    Exercise0,
    Exercise1,
    Exercise2,
    Exercise3,
    Exercise4,
    Exercise5,
    Exercise6,
    Exercise7,
    Exercise8,
    Exercise9,
    Exercise10,
    Exercise11,
    Exercise12,
    Exercise13,
    Exercise14,
    Exercise15,
    Exercise16,
    Exercise17,
    Exercise18,
    Exercise19,
    Exercise20,
    ],
    start=0,
)
__all__ = list(qvars)
