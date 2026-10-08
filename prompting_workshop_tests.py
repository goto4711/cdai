from learntools.core import *
import numpy as np
import pandas as pd


class Exercise1(CodingProblem):
    _vars = ['body', 'PROMPT']
    _hint = ("The body is a nested dictionary. Replace the first ___ with the variable PROMPT (no quotes: "
             "it is a variable that holds the text) and the second ___ with the number 1.0.")
    _solution = CS("""body = {
    'contents': [{'parts': [{'text': PROMPT}]}],
    'generationConfig': {'temperature': 1.0}
}""")

    def check(self, body, PROMPT):
        assert isinstance(body, dict), "body should be a dictionary."
        try:
            text = body['contents'][0]['parts'][0]['text']
        except (KeyError, IndexError, TypeError):
            raise AssertionError("Keep the structure of the body: {'contents': [{'parts': [{'text': ...}]}], ...}")
        if type(text).__name__ == 'PlaceholderValue':
            raise AssertionError("Replace the first ____ with the variable PROMPT.")
        assert text == PROMPT, ("The text in the body is not our prompt. Use the variable PROMPT, without quotes "
                                "around it.")
        try:
            t = body['generationConfig']['temperature']
        except (KeyError, TypeError):
            raise AssertionError("Keep 'generationConfig': {'temperature': ...} in the body.")
        if type(t).__name__ == 'PlaceholderValue':
            raise AssertionError("Replace the second ____ with the temperature, 1.0.")
        assert isinstance(t, (int, float)) and not isinstance(t, bool), "The temperature should be a number, e.g. 1.0."
        assert t == 1.0, "Set the temperature to 1.0, as in the pre-collected data."


class Exercise2(CodingProblem):
    _vars = ['answer', 'response']
    _hint = ("Go step by step and print each step: response['candidates'] is a list, "
             "response['candidates'][0] its first item, then ['content'], then ['parts'], "
             "then [0], then ['text'].")
    _solution = CS("answer = response['candidates'][0]['content']['parts'][0]['text']")

    def check(self, answer, response):
        parts = response['candidates'][0]['content']['parts']
        expected = parts[0].get('text')
        assert isinstance(answer, str), ("answer should be the text of the answer (a string). You are not there yet: "
                                         "go one step further along the path.")
        assert answer in (expected, ''.join(p.get('text', '') for p in parts)), "That is not the text of the model's answer. Follow the path in the hint."


class Exercise3(CodingProblem):
    _vars = ['tokens_in', 'tokens_answer', 'tokens_out', 'cost', 'response', 'PRICE_IN', 'PRICE_OUT']
    _hint = ("The token counts are in the dictionary usage = response['usageMetadata']: "
             "usage['promptTokenCount'] and usage['candidatesTokenCount'].")
    _solution = CS("""tokens_in = usage['promptTokenCount']
tokens_answer = usage['candidatesTokenCount']""")

    def check(self, tokens_in, tokens_answer, tokens_out, cost, response, PRICE_IN, PRICE_OUT):
        u = response['usageMetadata']
        assert tokens_in == u['promptTokenCount'], "tokens_in should be the value of 'promptTokenCount'."
        assert tokens_answer == u['candidatesTokenCount'], "tokens_answer should be the value of 'candidatesTokenCount'."
        assert tokens_out == tokens_answer + u.get('thoughtsTokenCount', 0), \
            "Keep the line that adds the thinking tokens as it was."
        assert abs(cost - (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6) < 1e-12, \
            "Keep the line that computes the cost as it was."


class Exercise4(ThoughtExperiment):
    _hint = ("The model does not look up an answer: it writes it word by word, each time choosing the next word "
             "from a list of likely words. What does the temperature do to that choice?")
    _solution = ("At temperature 1.0 the model samples each next word from its probability distribution, so every "
                 "run can take a different path and produce a different list. A lower temperature makes the answers "
                 "more similar, but not guaranteed identical (and Google advises against it for Gemini 3 models). "
                 "For research this means: one answer is one sample. You need several runs, and you must report "
                 "the model and the settings you used.")


class Exercise5(CodingProblem):
    _vars = ['events', 'missed']
    _hint = ("Select the column with events['year'] and call .between(1000, 2029) on it. "
             "It gives True or False for every row.")
    _solution = CS("events['old_method_sees'] = events['year'].between(1000, 2029)")

    def check(self, events, missed):
        assert 'old_method_sees' in events.columns, "Create the column events['old_method_sees']."
        expected = events['year'].between(1000, 2029)
        assert events['old_method_sees'].dtype == bool, "The column should contain True/False values: use .between()."
        assert (events['old_method_sees'] == expected).all(), \
            "Some rows are wrong: the old method sees the years from 1000 to 2029."
        exp_missed = 1 - expected.groupby(events['label']).mean()
        assert np.allclose(missed.sort_index().values, exp_missed.sort_index().values), \
            "Keep the line that computes missed as it was."


class Exercise6(CodingProblem):
    _vars = ['runs', 'within', 'between']
    _hint = ("en[i] and en[j] are dictionaries, one per answer. Compare their 'label' with ==: "
             "en[i]['label'] == en[j]['label'].")
    _solution = CS("if en[i]['label'] == en[j]['label']:")

    def check(self, runs, within, between):
        en = runs[runs['language'] == 'en'].to_dict('records')
        w, b = [], []
        for i in range(len(en)):
            for j in range(i + 1, len(en)):
                s = len(en[i]['year'] & en[j]['year']) / len(en[i]['year'] | en[j]['year'])
                (w if en[i]['label'] == en[j]['label'] else b).append(s)
        assert len(within) == len(w) and len(between) == len(b), \
            ("The pairs are not sorted correctly into within (same model) and between (different models). "
             "Compare the 'label' of the two answers.")
        assert np.isclose(np.mean(within), np.mean(w)), "within should contain the pairs from the same model."


class Exercise7(CodingProblem):
    _vars = ['my_regions', 'REGIONS']
    _hint = ("Write ten strings, one per event, e.g. ['Europe', 'East Asia', ...]. "
             "Copy the spelling from the list of regions printed above.")
    _solution = CS("my_regions = ['Europe', 'Global', ...]   # your own coding of the ten events")

    def check(self, my_regions, REGIONS):
        assert isinstance(my_regions, list), "my_regions should be a list."
        assert len(my_regions) == 10, f"my_regions should have 10 entries, one per event; it has {len(my_regions)}."
        wrong = [r for r in my_regions if r not in REGIONS]
        assert not wrong, f"Not in the list of regions: {wrong}. Copy the spelling exactly from REGIONS."


class Exercise8(CodingProblem):
    _vars = ['shares', 'events']
    _hint = ("pd.crosstab(rows, columns, normalize='index'): the rows are en_events['label'], "
             "the columns en_events['region'].")
    _solution = CS("shares = pd.crosstab(en_events['label'], en_events['region'], normalize='index')")

    def check(self, shares, events):
        en = events[events['language'] == 'en']
        exp = pd.crosstab(en['label'], en['region'], normalize='index')
        assert isinstance(shares, pd.DataFrame), "shares should be a table (DataFrame): use pd.crosstab()."
        if set(shares.index) == set(exp.columns):
            raise AssertionError("Rows and columns are swapped: models in the rows, regions in the columns.")
        assert set(shares.index) == set(exp.index), "Use one row per model (en_events['label'])."
        assert np.allclose(shares.sum(axis=1), 1), "Each row should add up to 1: add normalize='index'."
        try:
            got = shares.loc[exp.index, exp.columns].values
        except KeyError:
            raise AssertionError("Use the regions as columns: en_events['region'].")
        assert np.allclose(got, exp.values), "The shares are not right. Did you use the English events (en_events)?"


class Exercise9(CodingProblem):
    _vars = ['lang_shares', 'events']
    _hint = "Like the previous table, but with events['language'] for the rows and all events, not only the English ones."
    _solution = CS("lang_shares = pd.crosstab(events['language'], events['region'], normalize='index')")

    def check(self, lang_shares, events):
        exp = pd.crosstab(events['language'], events['region'], normalize='index')
        assert isinstance(lang_shares, pd.DataFrame), "lang_shares should be a table (DataFrame): use pd.crosstab()."
        assert set(lang_shares.index) == set(exp.index), "Use one row per language (events['language'])."
        try:
            got = lang_shares.loc[exp.index, exp.columns].values
        except KeyError:
            raise AssertionError("Use the regions as columns: events['region'].")
        assert np.allclose(got, exp.values), \
            "The shares are not right: use all events, the language as rows, and normalize='index'."


class Exercise10(CodingProblem):
    _vars = ['my_prompt', 'PROMPT']
    _hint = ("Replace ___ with your instruction, e.g. 'Give me a list of 20 important dates in the history of "
             "the internet.' Keep the two lines about the format.")
    _solution = CS("""my_prompt = '''Give me a list of 20 important dates in world history, as a history teacher in Lagos would choose them.
Format strictly as: YEAR | EVENT DESCRIPTION
One entry per line. No extra text.'''""")

    def check(self, my_prompt, PROMPT):
        assert isinstance(my_prompt, str), "my_prompt should be a string."
        assert '___' not in my_prompt, "Replace ___ with your own instruction."
        assert my_prompt.strip() != PROMPT.strip(), "Change something in the prompt."
        assert 'YEAR | EVENT' in my_prompt, "Keep the line about the format (YEAR | EVENT DESCRIPTION)."
        first = my_prompt.split('Format strictly')[0].strip()
        assert len(first) >= 15, "Write a full instruction before the format lines."


class Exercise11(CodingProblem):
    _vars = ['places', 'docs']
    _hint = "Replace ____ with ent.text: the text of each entity, e.g. 'France'."
    _solution = CS("places += [ent.text for ent in doc.ents if ent.label_ == 'GPE']")

    def check(self, places, docs):
        exp = [ent.text for doc in docs for ent in doc.ents if ent.label_ == 'GPE']
        assert isinstance(places, list), "places should be a list."
        assert not any(type(p).__name__ == 'PlaceholderValue' for p in places), "Replace ____ with ent.text."
        assert all(isinstance(p, str) for p in places), \
            "places should contain texts (strings): use ent.text, not ent."
        assert len(places) == len(exp), \
            f"places has {len(places)} entries, expected {len(exp)}. Keep only the entities with label 'GPE'."
        assert places == exp, "Some entries are not right: keep the text of each GPE entity."


qvars = bind_exercises(globals(), [
    Exercise1, Exercise2, Exercise3, Exercise4, Exercise5,
    Exercise6, Exercise7, Exercise8, Exercise9, Exercise10, Exercise11,
    ],
    start=1,
)
__all__ = list(qvars)
