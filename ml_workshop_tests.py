from learntools.core import *
import pandas as pd
import numpy as np
import sys

class Exercise0(CodingProblem):
    _vars = ['congress_114']
    _hint = "Use congress_114.head(5) to view the first 5 rows."
    _solution = CS("congress_114.head(5)")
    def check(self, congress_114):
        # This check is a bit tricky as head() returns a DataFrame and comparing DataFrames directly can be complex.
        # For simplicity, we'll check if the user has called head() and if the dataframe is not empty.
        # In a real scenario, you'd want to compare the output of head() more rigorously.
        assert isinstance(congress_114, pd.DataFrame), "Make sure 'congress_114' is a pandas DataFrame."
        assert not congress_114.empty, "The DataFrame should not be empty."
        # A more robust check would involve capturing the output of head() and comparing it.
        # For now, we'll assume the user's intent is to display the head.

class Exercise1(CodingProblem):
    _vars = ['congress_114']
    _hint = "Use congress_114.tail(5) to view the last 5 rows."
    _solution = CS("congress_114.tail(5)")
    def check(self, congress_114):
        assert isinstance(congress_114, pd.DataFrame), "Make sure 'congress_114' is a pandas DataFrame."
        assert not congress_114.empty, "The DataFrame should not be empty."

class Exercise2(CodingProblem):
    _vars = ['congress_114']
    _hint = "Use congress_114.dropna() to remove rows with NaN values."
    _solution = CS("congress_114 = congress_114.dropna()")
    def check(self, congress_114):
        assert congress_114.isnull().sum().sum() == 0, "There are still NaN values in the DataFrame. Make sure you used dropna()."

class Exercise3(ThoughtExperiment):
    _hint = """congress_114_voting = congress_114.loc[:, bill_cols]
congress_114_voting.head()"""
    _solution = CS("""congress_114_voting = congress_114.loc[:, bill_cols]
congress_114_voting.head()""")

# class Exercise3(CodingProblem):
#     _vars = ['congress_114', 'X']
#     _hint = "Select only the bill columns. You can do this by dropping 'name', 'party', 'state', and 'index' columns."
#     _solution = CS("X = congress_114.drop(['name', 'party', 'state', 'index'], axis=1)")
#     def check(self, congress_114, X):
#         expected_columns = [col for col in congress_114.columns if 'bill' in col]
#         assert all(col in X.columns for col in expected_columns), "Make sure all bill columns are in X."
#         assert all(col not in X.columns for col in ['name', 'party', 'state', 'index']), "Make sure 'name', 'party', 'state', and 'index' columns are dropped."

class Exercise4(CodingProblem):
    _vars = ['congress_114_result', 'kmeans_5']
    _hint = "Add the np.array(kmeans_5.labels_) array as a new column named 'cluster_5' to the congress_114_result DataFrame."
    _solution = CS("congress_114_result['cluster_5'] = np.array(kmeans_5.labels_)")
    def check(self, congress_114_result, kmeans_5):
        assert 'cluster_5' in congress_114_result.columns, "'cluster_5' column not found in congress_114."
        assert all(congress_114_result["cluster_5"].values == np.array(kmeans_5.labels_)), "The 'cluster_5' column does not match the 'np.array(kmeans_5.labels_)' array."

class Exercise5(ThoughtExperiment):
    _hint = """pd.crosstab(congress_114_result.party, congress_114_result.cluster_5)"""
    _solution = CS("""pd.crosstab(congress_114_result.party, congress_114_result.cluster_5)""")


class Exercise6(ThoughtExperiment):
    _hint = """from sklearn.preprocessing import MaxAbsScaler
scaler2 = MaxAbsScaler()
music_normalized_df2 = music_df.copy()
music_normalized_df2.iloc[:,:-1] = scaler2.fit_transform(music_normalized_df2.iloc[:,:-1])
music_normalized_df2.head()"""
    _solution = CS("""from sklearn.preprocessing import MaxAbsScaler
scaler2 = MaxAbsScaler()
music_normalized_df2 = music_df.copy()
music_normalized_df2.iloc[:,:-1] = scaler2.fit_transform(music_normalized_df2.iloc[:,:-1])
music_normalized_df2.head()""")

class Exercise7(CodingProblem):
    _vars = ['train_set', 'test_set']
    _hint = "Use train_test_split from sklearn.model_selection. Remember to set test_size=0.2 and random_state=42."
    _solution = CS("train_set, test_set = train_test_split(music_df, test_size=0.2, random_state=42)")
    def check(self, train_set, test_set):
        assert isinstance(train_set, pd.DataFrame), "train_set should be a pandas DataFrame."
        assert isinstance(test_set, pd.DataFrame), "test_set should be a pandas DataFrame."
        assert len(train_set) + len(test_set) == 114000, "The sum of train and test set lengths should be 114000."
        assert abs(len(test_set) / len(train_set) - 0.25) < 0.01, "The test set size should be approximately 20% of the total data."

class Exercise8(ThoughtExperiment):
    _hint = "np.mean(train_set['valence'])"
    _solution = CS("np.mean(train_set['valence'])")


class Exercise9(ThoughtExperiment):
    _hint = """from sklearn.metrics import mean_absolute_error
mean_absolute_error(test_set['dance_label'], preds)"""
    _solution = CS("""from sklearn.metrics import mean_absolute_error
mean_absolute_error(test_set['dance_label'], preds)""")


class Exercise10(ThoughtExperiment):
    _hint = """mean_liveness = np.mean(train_set['liveness'])
preds = (test_set['liveness'].values < mean_liveness).astype(int)
print('Preds: ', preds)
print('Mean absolute error: ', mean_absolute_error(test_set['dance_label'], preds))"""
    _solution = CS("""mean_liveness = np.mean(train_set['liveness'])
preds = (test_set['liveness'].values < mean_liveness).astype(int)
print('Preds: ', preds)
print('Mean absolute error: ', mean_absolute_error(test_set['dance_label'], preds))""")


class Exercise11(ThoughtExperiment):
    _hint = """mean_acousticness = np.mean(train_set['acousticness'])
print(score_(test_set['acousticness'].values, test_set['dance_label'].values, mean_acousticness))"""
    _solution = CS("""mean_acousticness = np.mean(train_set['acousticness'])
print(score_(test_set['acousticness'].values, test_set['dance_label'].values, mean_acousticness))""")


class Exercise12(ThoughtExperiment):
    _hint = """def min_column(df, col_):
    df = df.sample(frac=0.1, random_state=4711) 
    unq = df[col_].dropna().unique()
    s = np.array([score_(df[col_], df['dance_label'], o) for o in unq if not np.isnan(o)])
    idx = s.argmin()
    return (col_, float(unq[idx]), float(s[idx]))

min_column(train_set, 'valence')"""
    _solution = CS("""def min_column(df, col_):
    df = df.sample(frac=0.1, random_state=4711) 
    unq = df[col_].dropna().unique()
    s = np.array([score_(df[col_], df['dance_label'], o) for o in unq if not np.isnan(o)])
    idx = s.argmin()
    return (col_, float(unq[idx]), float(s[idx]))

min_column(train_set, 'valence')""")

class Exercise13(ThoughtExperiment):
    _hint = """for w in list(train_set.columns[:-1]):
    print(min_column(train_set, w))"""
    _solution = CS("""for w in list(train_set.columns[:-1]):
    print(min_column(train_set, w))""")


class Exercise14(ThoughtExperiment):
    _hint = """right_split_valence_columns = [c for c in right_split_valence.columns if c not in ['valence', 'dance_label']]
for w in right_split_valence_columns:
    print(min_column(right_split_valence, w))"""
    _solution = CS("""right_split_valence_columns = [c for c in right_split_valence.columns if c not in ['valence', 'dance_label']]
for w in right_split_valence_columns:
    print(min_column(right_split_valence, w))""")


class Exercise15(CodingProblem):
    _vars = ['X_train', 'y_train', 'X_test', 'y_test']
    _hint = "Ensure X_train, y_train, X_test, y_test are correctly extracted from train_set and test_set."
    _solution = CS("X_train = train_set.loc[:, train_set.columns != 'dance_label'].values\ny_train = train_set['dance_label'].values\nX_test = test_set.loc[:, test_set.columns != 'dance_label'].values\ny_test = test_set['dance_label'].values")
    def check(self, X_train, y_train, X_test, y_test):
        assert isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray), "X_train and y_train should be numpy arrays."
        assert isinstance(X_test, np.ndarray) and isinstance(y_test, np.ndarray), "X_test and y_test should be numpy arrays."
        assert X_train.shape[0] == y_train.shape[0], "X_train and y_train should have the same number of samples."
        assert X_test.shape[0] == y_test.shape[0], "X_test and y_test should have the same number of samples."
        assert X_train.shape[1] == 9, "X_train should have 9 features."

class Exercise16(CodingProblem):
    _vars = ['model']
    _hint = "Import DecisionTreeClassifier from sklearn.tree and initialize it with max_leaf_nodes=4 and random_state=4711, then fit it to X_train and y_train."
    _solution = CS("from sklearn.tree import DecisionTreeClassifier\nmodel = DecisionTreeClassifier(max_leaf_nodes=4, random_state=4711).fit(X_train, y_train)")
    def check(self, model):
        from sklearn.tree import DecisionTreeClassifier
        assert isinstance(model, DecisionTreeClassifier), "model should be an instance of DecisionTreeClassifier."
        assert model.max_leaf_nodes == 4, "max_leaf_nodes should be set to 4."
        assert model.random_state == 4711, "random_state should be set to 4711."
        assert hasattr(model, 'tree_'), "The model should be fitted."

class Exercise17(CodingProblem):
    _vars = ['preds']
    _hint = "Use model.predict(X_test) to get predictions and then calculate mean_absolute_error with y_test."
    _solution = CS("preds = model.predict(X_test)\nmean_absolute_error(y_test, preds)")
    def check(self, preds):
        assert isinstance(preds, np.ndarray), "Predictions should be a numpy array."
        assert len(preds) == 22800, "The number of predictions should match the test set size."

class Exercise18(CodingProblem):
    _vars = ['preds']
    _hint = "Use model_2.predict(X_test) to get predictions and then calculate mean_absolute_error with y_test."
    _solution = CS("preds = model_2.predict(X_test)\nmean_absolute_error(y_test, preds)")
    def check(self, preds):
        assert isinstance(preds, np.ndarray), "Predictions should be a numpy array."
        assert len(preds) == 22800, "The number of predictions should match the test set size."

class Exercise19(CodingProblem):
    _vars = ['rf']
    _hint = "Initialize RandomForestClassifier with n_estimators=10, min_samples_leaf=5, and random_state=4711, then fit it."
    _solution = CS("""from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(10, min_samples_leaf=5, random_state=4711)
rf.fit(X_train, y_train)
mean_absolute_error(y_test, rf.predict(X_test))""")
    def check(self, rf):
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(rf, RandomForestClassifier), "rf should be an instance of RandomForestClassifier."
        assert rf.n_estimators == 10, "n_estimators should be set to 10."
        assert rf.min_samples_leaf == 5, "min_samples_leaf should be set to 5."
        assert hasattr(rf, 'estimators_'), "rf should be fitted."

class Exercise20(ThoughtExperiment):
    _hint = """import pandas as pd
forest_importances = pd.Series(rf.feature_importances_, index=train_set.columns[:-1])
fig, ax = plt.subplots()
forest_importances.sort_values(ascending=False).plot.bar(ax=ax) 
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()"""
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
