from learntools.core import *
import numpy as np
import pandas as pd


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
    _vars = ['my_regions', 'REGIONS']
    _hint = ("Write ten strings, one per event, e.g. ['Europe', 'East Asia', ...]. "
             "Copy the spelling from the list of regions printed above.")
    _solution = CS("my_regions = ['Europe', 'Global', ...]   # your own coding of the ten events")

    def check(self, my_regions, REGIONS):
        assert isinstance(my_regions, list), "my_regions should be a list."
        assert len(my_regions) == 10, f"my_regions should have 10 entries, one per event; it has {len(my_regions)}."
        wrong = [r for r in my_regions if r not in REGIONS]
        assert not wrong, f"Not in the list of regions: {wrong}. Copy the spelling exactly from REGIONS."


class Exercise4(CodingProblem):
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


qvars = bind_exercises(globals(), [Exercise1, Exercise2, Exercise3, Exercise4], start=1)
__all__ = list(qvars)
