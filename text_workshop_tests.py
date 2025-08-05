from learntools.core import *
import numpy as np
import pandas as pd 


class Exercise0(ThoughtExperiment):
    _hint = ("Explore the structure of your dataset by examining its features. "
             "The .features attribute shows you the data types and structure of each column. "
             "This is essential for understanding what kind of data you're working with.")
    _solution = CS("""train_ds.features""")


class Exercise1(ThoughtExperiment):
    _hint = ("Examine your dataset structure and sample data to understand the content. "
             "First print the dataset overview, then look at the first 5 examples from the training split. "
             "This helps you understand the data format and content before processing.")
    _solution = CS("""print(ds)
print(ds['train'][:5])""")


class Exercise2(CodingProblem):
    _vars = ['df_hs']  
    _hint = ("Create human-readable labels for your binary classification task. "
             "Use np.where() to map numeric labels (0/1) to descriptive names. "
             "Syntax: np.where(condition, value_if_true, value_if_false). "
             "This makes your data analysis more interpretable.")
    _solution = CS("""df_hs['label_name'] = np.where(df_hs['label'] == 0, 'Non-hateful', 'Hateful')
df_hs.head()""")
    
    def check(self, df_hs):
        if not isinstance(df_hs, pd.DataFrame):
            raise AssertionError("Make sure 'df_hs' is a pandas DataFrame.")
        
        if 'label_name' not in df_hs.columns:
            raise AssertionError("Column 'label_name' not found. Make sure you created the new column with descriptive labels.")
        
        if len(df_hs['label_name']) == 0:
            raise AssertionError("The 'label_name' column is empty. Make sure you used np.where() to populate it.")
        
        # Check that labels make sense
        unique_labels = set(df_hs['label_name'].unique())
        expected_labels = {'Non-hateful', 'Hateful'}
        if not unique_labels.issubset(expected_labels):
            raise AssertionError(f"Expected labels to be 'Non-hateful' and 'Hateful', but found {unique_labels}.")


class Exercise3(ThoughtExperiment):
    _hint = ("Visualize the class distribution in your hate speech dataset using a bar plot. "
             "Use value_counts() to count occurrences of each class, then create a horizontal bar plot. "
             "Class imbalance is common in hate speech detection and affects model performance.")
    _solution = CS("""import matplotlib.pyplot as plt
df_hs["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()""")


class Exercise4(ThoughtExperiment):
    _hint = ("Analyze text length patterns across different classes using box plots. "
             "First, create a 'Words Per Tweet' column by splitting text and counting words. "
             "Then use boxplot() to compare word count distributions between hateful and non-hateful tweets. "
             "Text length can be a useful feature for classification.")
    _solution = CS("""df_hs["Words Per Tweet"] = df_hs["tweet"].str.split().apply(len)
df_hs.boxplot("Words Per Tweet", by="label_name", grid=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()""")


class Exercise5(CodingProblem):
    _vars = ['tokenized_text']
    _hint = ("Convert tokens to numerical IDs using a token-to-index mapping. "
             "This is a fundamental step in NLP: converting text tokens to numbers that models can process. "
             "Use list comprehension to map each token to its corresponding ID: [token2idx[token] for token in tokens]")
    _solution = CS("input_ids = [token2idx[token] for token in tokenized_text]")
    
    def check(self, tokenized_text):
        if not tokenized_text or len(tokenized_text) == 0:
            raise AssertionError("tokenized_text appears to be empty. Make sure you've tokenized your text properly.")
        
        if not isinstance(tokenized_text, (list, tuple)):
            raise AssertionError("tokenized_text should be a list or tuple of tokens.")


class Exercise6(ThoughtExperiment):
    _hint = ("Use a pre-trained tokenizer to encode text into the format expected by transformer models. "
             "Modern tokenizers handle subword tokenization, special tokens, and attention masks automatically. "
             "Simply call tokenizer(text) to get input_ids, attention_mask, and other required tensors.")
    _solution = CS("""encoded_text = tokenizer(text)
print(encoded_text)""")


class Exercise7(ThoughtExperiment):
    _hint = ("Convert tokenized text back to human-readable format using the tokenizer. "
             "The convert_tokens_to_string() method reconstructs text from tokens, "
             "handling subword pieces and special formatting. This is useful for debugging and understanding tokenization.")
    _solution = CS("""tokenizer.convert_tokens_to_string(tokens)""")


class Exercise8(CodingProblem):
    _vars = ['y_preds']
    _hint = ("Examine the first 10 predictions from your model to understand the output format. "
             "Model predictions are typically arrays of probabilities or class indices. "
             "Looking at a sample helps verify your model is working correctly.")
    _solution = CS("y_preds[:10]")
    
    def check(self, y_preds):
        if len(y_preds) <= 9:
            raise AssertionError(f"y_preds should have at least 10 elements, but has {len(y_preds)}. "
                               "Make sure your model generated predictions for the full dataset.")
        
        if not isinstance(y_preds, (np.ndarray, list)):
            raise AssertionError("y_preds should be a numpy array or list of predictions.")


class Exercise9(ThoughtExperiment):
    _hint = ("Create a confusion matrix to evaluate your hate speech classification model. "
             "Import confusion matrix tools from sklearn.metrics, extract true labels from your test data, "
             "then create and display the confusion matrix. This shows true positives, false positives, etc.")
    _solution = CS("""from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_test = np.array(hatespeech_encoded_test_small["labels"])
cm = confusion_matrix(y_test, y_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()""")


class Exercise10(CodingProblem):
    _vars = ['df_pat']
    _hint = ("Load the patient data CSV file and explore its structure. "
             "Use pd.read_csv() to load the data, then use .info() to see column types and null values, "
             "and .head() to examine the first few rows. Understanding your data structure is crucial for NLP tasks.")
    _solution = CS("""df_pat = pd.read_csv('./uspppm-data/train.csv')
print(df_pat.info())
print(df_pat.head(3))""")
    
    def check(self, df_pat):
        if not isinstance(df_pat, pd.DataFrame):
            raise AssertionError("df_pat should be a pandas DataFrame. Make sure you used pd.read_csv() correctly.")
        
        if df_pat.empty:
            raise AssertionError("The DataFrame df_pat is empty. Check that the CSV file exists and loaded correctly.")
        
        # Check for expected structure (common columns in medical/patient data)
        if len(df_pat.columns) < 2:
            raise AssertionError("The DataFrame seems to have very few columns. Make sure the CSV loaded properly.")


class Exercise11(ThoughtExperiment):
    _hint = ("Analyze the distribution of different contexts in your patient data. "
             "Use value_counts() on the 'context' column to see how many examples exist for each context type. "
             "Understanding context distribution helps with model training and evaluation strategies.")
    _solution = CS("""df_pat['context'].value_counts()""")


class Exercise12(ThoughtExperiment):
    _hint = ("Initialize a tokenizer for your specific model checkpoint. "
             "AutoTokenizer.from_pretrained() automatically loads the correct tokenizer for your model. "
             "Each pre-trained model has its own tokenizer that matches the vocabulary used during training.")
    _solution = CS("""tokenizer_pat = AutoTokenizer.from_pretrained(checkpoint_pat)""")


class Exercise13(ThoughtExperiment):
    _hint = ("Apply tokenization to your entire dataset using the map() function. "
             "Use batched=True for efficient processing of large datasets. "
             "The map() function applies your tokenizer function to all examples, creating the input format needed for transformer models.")
    _solution = CS("""ds_pat_encoded = ds_pat.map(tokenizer_pat_func, batched=True)
ds_pat_encoded""")


class Exercise14(ThoughtExperiment):
    _hint = ("Compare the original text input with its tokenized representation. "
             "Print both the original 'input' text and the tokenized 'input_ids' to understand "
             "how your text gets converted to numerical format that the model can process.")
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
