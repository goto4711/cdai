from learntools.core import *
import numpy as np
import pandas as pd 


class Exercise0(ThoughtExperiment):
    _hint = """train_ds.features"""
    _solution = CS("""train_ds.features""")

class Exercise1(ThoughtExperiment):
    _hint = """print(ds)
print(ds['train'][:5])"""
    _solution = CS("""print(ds)
print(ds['train'][:5])""")

class Exercise2(CodingProblem):
    _vars = ['df_hs']
    _hint = """df_hs['label_name'] = np.where(df_hs['label'] == 0, 'Non-hateful', 'Hateful')
df_hs.head()"""
    _solution = CS(""""df_hs['label_name'] = np.where(df_hs['label'] == 0, 'Non-hateful', 'Hateful')
df_hs.head()""")
    def check(self, df_hs):
        assert isinstance(df_hs, pd.DataFrame), "Make sure 'df_hs' is a pandas DataFrame."
        assert len(list(df_hs.label_name))>0, f"\n💡 Hint: {self._hint}"

class Exercise3(ThoughtExperiment):
    _hint = """import matplotlib.pyplot as plt
df_hs["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()"""
    _solution = CS("""import matplotlib.pyplot as plt
df_hs["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()""")

class Exercise4(ThoughtExperiment):
    _hint = """df_hs["Words Per Tweet"] = df_hs["tweet"].str.split().apply(len)
df_hs.boxplot("Words Per Tweet", by="label_name", grid=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()"""
    _solution = CS("""df_hs["Words Per Tweet"] = df_hs["tweet"].str.split().apply(len)
df_hs.boxplot("Words Per Tweet", by="label_name", grid=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()""")

class Exercise5(CodingProblem):
    _vars = ['tokenized_text']
    _hint = """input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)"""
    _solution = CS("input_ids = [token2idx[token] for token in tokenized_text]")
    def check(self, tokenized_text):
        assert len(tokenized_text)>0, f"\n💡 Hint: {self._hint}"

class Exercise6(ThoughtExperiment):
    _hint = """encoded_text = tokenizer(text)
print(encoded_text)"""
    _solution = CS("""encoded_text = tokenizer(text)
print(encoded_text)""")

class Exercise7(ThoughtExperiment):
    _hint = """tokenizer.convert_tokens_to_string(tokens)"""
    _solution = CS("""tokenizer.convert_tokens_to_string(tokens)""")

class Exercise8(CodingProblem):
    _vars = ['y_preds']
    _hint = """y_preds[:10]"""
    _solution = CS("iy_preds[:10]")
    def check(self, y_preds):
        assert len(y_preds) > 9, f"\n💡 Hint: {self._hint}"

class Exercise9(ThoughtExperiment):
    _hint = """from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_test = np.array(hatespeech_encoded_test_small["labels"])
cm = confusion_matrix(y_test, y_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()"""
    _solution = CS("""from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_test = np.array(hatespeech_encoded_test_small["labels"])
cm = confusion_matrix(y_test, y_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()""")


class Exercise10(CodingProblem):
    _vars = ['df_pat']
    _hint = """df_pat = pd.read_csv('./uspppm-data/train.csv')
print(df_pat.info())
print(df_pat.head(3))"""
    _solution = CS("""df_pat = pd.read_csv('./uspppm-data/train.csv')
print(df_pat.info())
print(df_pat.head(3))""")
    def check(self, df_pat):
        assert isinstance(df_pat, pd.DataFrame), "Make sure 'df_pat' is a pandas DataFrame."
        assert not df_pat.empty, "The DataFrame df_pat should not be empty."

class Exercise11(ThoughtExperiment):
    _hint = """df_pat['context'].value_counts()"""
    _solution = CS("""df_pat['context'].value_counts()""")

class Exercise12(ThoughtExperiment):
    _hint = """tokenizer_pat = AutoTokenizer.from_pretrained(checkpoint_pat)"""
    _solution = CS("""tokenizer_pat = AutoTokenizer.from_pretrained(checkpoint_pat)""")

class Exercise13(ThoughtExperiment):
    _hint = """ds_pat_encoded = ds_pat.map(tokenizer_pat_func, batched=True)
ds_pat_encoded"""
    _solution = CS("""ds_pat_encoded = ds_pat.map(tokenizer_pat_func, batched=True)
ds_pat_encoded""")

class Exercise14(ThoughtExperiment):
    _hint = """print('input: ', ds_pat_encoded[0]['input'])
print('input_ids: ', ds_pat_encoded[0]['input_ids'])"""
    _solution = CS("""print('input: ', ds_pat_encoded[0]['input'])
print('input_ids: ', ds_pat_encoded[0]['input_ids'])""")



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
    ],
    start=0,
)
__all__ = list(qvars)
