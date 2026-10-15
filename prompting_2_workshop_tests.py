from learntools.core import *
import numpy as np
import pandas as pd


def _placeholder(x):
    return type(x).__name__ == 'PlaceholderValue'


class Exercise1(CodingProblem):
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


class Exercise2(ThoughtExperiment):
    _hint = ("The model does not look up an answer: it writes it word by word, each time choosing the next word "
             "from a list of likely words. What does the temperature do to that choice?")
    _solution = ("At temperature 1.0 the model samples each next word from its probability distribution, so every "
                 "run can take a different path and produce a different list. A lower temperature makes the answers "
                 "more similar, but not guaranteed identical (and Google advises against it for Gemini 3 models). "
                 "For research this means: one answer is one sample. You need several runs, and you must report "
                 "the model and the settings you used.")


class Exercise3(CodingProblem):
    _vars = ['latest', 'probes']
    _hint = ("Three steps in one line: probes[probes['knows']] keeps the rows where knows is True; "
             ".groupby('label') makes one group per model; ['event_date'].max() takes the latest date.")
    _solution = CS("latest = probes[probes['knows']].groupby('label')['event_date'].max()")

    def check(self, latest, probes):
        exp = probes[probes['knows']].groupby('label')['event_date'].max()
        assert isinstance(latest, pd.Series), "latest should be one value per model: use .groupby('label')['event_date'].max()."
        assert set(latest.index) == set(exp.index), \
            "latest should have one row per model that knows at least one event. Did you keep only the rows where knows is True?"
        assert (latest.sort_index() == exp.sort_index()).all(), \
            "Some dates are not right. Use .max() on the column event_date."


class Exercise4(CodingProblem):
    _vars = ['forecasts']
    _hint = "Write (forecasts['probability'] - forecasts['outcome']) ** 2"
    _solution = CS("forecasts['brier'] = (forecasts['probability'] - forecasts['outcome']) ** 2")

    def check(self, forecasts):
        assert 'brier' in forecasts.columns, "Create the column forecasts['brier']."
        assert not _placeholder(forecasts['brier'].iloc[0]), "Replace ____ with the formula."
        exp = (forecasts['probability'] - forecasts['outcome']) ** 2
        assert np.allclose(forecasts['brier'].astype(float), exp), \
            "Not the Brier score yet: subtract the outcome from the probability, then square with ** 2."


qvars = bind_exercises(globals(), [Exercise1, Exercise2, Exercise3, Exercise4], start=1)
__all__ = list(qvars)
