from learntools.core import *
import numpy as np
import pandas as pd


def _placeholder(x):
    return type(x).__name__ == 'PlaceholderValue'


class Exercise12(CodingProblem):
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


class Exercise13(CodingProblem):
    _vars = ['forecasts']
    _hint = "Write (forecasts['probability'] - forecasts['outcome']) ** 2"
    _solution = CS("forecasts['brier'] = (forecasts['probability'] - forecasts['outcome']) ** 2")

    def check(self, forecasts):
        assert 'brier' in forecasts.columns, "Create the column forecasts['brier']."
        assert not _placeholder(forecasts['brier'].iloc[0]), "Replace ____ with the formula."
        exp = (forecasts['probability'] - forecasts['outcome']) ** 2
        assert np.allclose(forecasts['brier'].astype(float), exp), \
            "Not the Brier score yet: subtract the outcome from the probability, then square with ** 2."


class Exercise14(CodingProblem):
    _vars = ['fair', 'forecasts']
    _hint = "Put the condition in square brackets: forecasts[~forecasts['leak']]"
    _solution = CS("fair = forecasts[~forecasts['leak']]")

    def check(self, fair, forecasts):
        assert isinstance(fair, pd.DataFrame), "fair should be a table: select rows of forecasts with [ ... ]."
        exp = forecasts[~forecasts['leak']]
        if len(fair) == int(forecasts['leak'].sum()):
            raise AssertionError("You kept the leaks: use ~ (not) or == False.")
        assert len(fair) == len(exp) and not fair['leak'].any(), "fair should contain exactly the rows where leak is False."


class Exercise15(CodingProblem):
    _vars = ['by_break', 'fair']
    _hint = "Replace ____ with the name of the column, in quotes: 'break'."
    _solution = CS("by_break = fair.groupby(['label', 'break'])['brier'].mean().unstack()")

    def check(self, by_break, fair):
        exp = fair.groupby(['label', 'break'])['brier'].mean().unstack()
        assert isinstance(by_break, pd.DataFrame), "Keep .unstack() at the end, so that you get a table."
        assert set(by_break.index) == set(exp.index) and set(by_break.columns) == set(exp.columns), \
            "Group by 'label' and 'break' (models in the rows, True/False in the columns)."
        got = by_break.reindex(index=exp.index, columns=exp.columns)    # (.loc would read [False, True] as a mask)
        assert np.allclose(got.values.astype(float), exp.values.astype(float), equal_nan=True), \
            "The averages are not right: use the column brier and .mean()."


class Exercise17(CodingProblem):
    _vars = ['spread', 'fair']
    _hint = "fair.groupby(['label', 'id'])['probability'].std()"
    _solution = CS("spread = fair.groupby(['label', 'id'])['probability'].std()")

    def check(self, spread, fair):
        exp = fair.groupby(['label', 'id'])['probability'].std()
        assert isinstance(spread, pd.Series), "spread should be one value per model and question: group by ['label', 'id']."
        assert len(spread) == len(exp), "Group by both 'label' and 'id'."
        assert np.allclose(spread.sort_index().values, exp.sort_index().values, equal_nan=True), \
            "Use .std() on the column probability."


class Exercise16(CodingProblem):
    _vars = ['continuity_score', 'continuity_words']
    _hint = "The condition is the same as in the lecture's hate_score_(): word in continuity_words"
    _solution = CS("""    if word in continuity_words:""")

    def check(self, continuity_score, continuity_words):
        from nltk.tokenize import word_tokenize
        def ref(text):
            return sum(1 for w in word_tokenize(text) if w.lower() in continuity_words)
        tests = ['Historically, such regimes remain in power; a collapse is unlikely.',
                 'The strikes make a rapid change likely.', 'Usually usually USUALLY.', '']
        for x in tests:
            try:
                got = continuity_score(x)
            except Exception as e:
                raise AssertionError(f"continuity_score() stops with an error: {e}")
            assert got == ref(x), (f"continuity_score({x!r}) gives {got}, but it should be {ref(x)}. "
                                   "Count a word when it is in continuity_words.")


qvars = bind_exercises(globals(), [Exercise12, Exercise13, Exercise14, Exercise15, Exercise16, Exercise17], start=12)
__all__ = list(qvars)
