from learntools.core import *
#from learntools.core.problem import injected
import numpy as np

class Exercise0(ThoughtExperiment):
    _hint = ("5 * (4 - 3)")
    _solution = CS("5 * (4 - 3)")


class Exercise1(CodingProblem):
    _vars = ['my_classes', 'theory_classes', 'python_classes']
    _hint = "Define my_classes = theory_classes + python_classes."
    _solution = CS('my_classes = theory_classes + python_classes')
    def check(self, my_classes, theory_classes, python_classes):
        assert my_classes == theory_classes + python_classes, f"\n💡 Hint: {self._hint}"
            

class Exercise2(CodingProblem):
    _var = 'my_days'
    _hint = """my_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            my_days.append('Saturday')
            my_days.append('Sunday')"""
    _solution = CS("""my_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            my_days.append('Saturday')
            my_days.append('Sunday')""")
    def check(self, my_days):
        assert my_days == ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], f"\n💡 Hint: {self._hint}"


class Exercise3(ThoughtExperiment):
    _hint = """print(len(my_numbers)
print(len(my_days))"""
    _solution = CS("""print(len(my_numbers)
print(len(my_days)""")
    
class Exercise4(CodingProblem):
    _var = 'linkedin_list'
    _hint = """sum(linkedin_list)"""
    _solution = CS("sum(linkedin_list)")
    def check(self, linkedin_list):
        assert sum(linkedin_list) > 70, f"\n💡 Hint: {self._hint}"
    
class Exercise5(CodingProblem):
    _vars = ['facebook_list', 'facebook_monday']
    _hint = "facebook_monday = facebook_list[0]"
    _solution = CS("facebook_monday = facebook_list[0]")
    def check(self, facebook_list, facebook_monday):
        assert (facebook_monday == facebook_list[0]), f"\n💡 Hint: {self._hint}"


class Exercise6(CodingProblem):
    _vars = ['facebook_list', 'facebook']
    _hint = "facebook = np.array(facebook_list)"
    _solution = CS("facebook = np.array(facebook_list)")
    def check(self, facebook_list, facebook):
        assert ((facebook == np.array(facebook_list)).all()), f"\n💡 Hint: {self._hint}"


class Exercise7(ThoughtExperiment):
    _hint = ('views == 13')
    _solution = CS("views == 13")

class Exercise8(EqualityCheckProblem):
    _var = 'num_views'
    _expected = 14
    _solution = CS(
"""if (num_views > 13):
    print('You are very popular!')
""")

class Exercise9(ThoughtExperiment):
    _hint = """print(for i in range(3):
    print(i**2)"""
    _solution = CS("""for i in range(3):
    print(i**2)""")


class Exercise10(CodingProblem):
    _vars = ['facebook', 'linkedin']
    _hint = "np.mean(facebook + linkedin)"
    _solution = CS("np.mean(facebook + linkedin)")
    def check(self, facebook, linkedin):
        assert (np.mean(facebook + linkedin) > 22) and (np.mean(facebook + linkedin) < 22.5), f"\n💡 Hint: {self._hint}"

class Exercise11(ThoughtExperiment):
    _hint = """
    for i in range(len(facebook_list)):
        if (facebook_list[i] > linkedin_list[i]):
            print('Facebook wins.')
        else:
            print('LinkedIn wins.')
    """
    _solution = CS("""
    for i in range(len(facebook_list)):
        if (facebook_list[i] > linkedin_list[i]):
            print('Facebook wins.')
        else:
            print('LinkedIn wins.')
    """)


class Exercise12(CodingProblem):
    _var = 'min_freq'
    _hint = "min_freq = np.min(freq)"
    _solution = CS("min_freq = np.min(freq)")
    def check(self, min_freq):
        assert min_freq == 0, f"\n💡 Hint: {self._hint}"


class Exercise13(ThoughtExperiment):
    _hint = """
    deathpenalty_array[freq == min_freq]
    """
    _solution = CS("""
    deathpenalty_array[freq == min_freq]
    """)

class Exercise14(ThoughtExperiment):
    _hint = """
    n_black_deathpenalty / (n_black_deathpenalty + n_white_deathpenalty)
    """
    _solution = CS("""
    n_black_deathpenalty / (n_black_deathpenalty + n_white_deathpenalty)
    """)


class Exercise15(ThoughtExperiment):
    _hint = """
    white_defendant_and_black_victim_and_deathpenalty = (defendantrace == 1) & (victimrace == 0) & (deathpenalty == 1)
    np.sum(freq[white_defendant_and_black_victim_and_deathpenalty]) / n_white_deathpenalty
    """
    _solution = CS("""
    white_defendant_and_black_victim_and_deathpenalty = (defendantrace == 1) & (victimrace == 0) & (deathpenalty == 1)
    np.sum(freq[white_defendant_and_black_victim_and_deathpenalty]) / n_white_deathpenalty
    """)



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
    ],
    start=0,
    )
__all__ = list(qvars)
