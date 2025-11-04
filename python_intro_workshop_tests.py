from learntools.core import *
import numpy as np

class Exercise0(ThoughtExperiment):
    _hint = ("Think about the order of operations (https://en.wikipedia.org/wiki/Order_of_operations). What happens when you multiply 5 by the result of (4 - 3)?")
    _solution = CS("5 * (4 - 3)")


class Exercise1(CodingProblem):
    _vars = ['my_classes', 'theory_classes', 'python_classes']
    _hint = ("You need to combine the two lists using the + operator. "
             "Remember that list concatenation works as: list1 + list2")
    _solution = CS('my_classes = theory_classes + python_classes')
    
    def check(self, my_classes, theory_classes, python_classes):
        
        expected = theory_classes + python_classes
        if my_classes != expected:
            raise AssertionError(f"Expected my_classes to equal {expected}, but got {my_classes}. "
                               f"Make sure you're adding theory_classes and python_classes together.")
            

class Exercise2(CodingProblem):
    _var = 'my_days'
    _hint = ("Start by creating a list with the weekdays, then use the .append() method to add weekend days. "
             "Remember: my_list.append('item') adds 'item' to the end of my_list.")
    _solution = CS("""my_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
my_days.append('Saturday')
my_days.append('Sunday')""")
    
    def check(self, my_days):
        expected = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if not isinstance(my_days, list):
            raise AssertionError("Expected my_days to be a list.")
        
        if len(my_days) != 7:
            raise AssertionError(f"Expected 7 days, but got {len(my_days)}. "
                               "Make sure you have all weekdays plus Saturday and Sunday.")
        
        if my_days != expected:
            raise AssertionError(f"Expected {expected}, but got {my_days}. "
                               "Check the spelling and order of your days.")


class Exercise3(ThoughtExperiment):
    _hint = ("Use the len() function to find the length of a list. "
             "For example: len(my_list) returns the number of items in my_list. "
             "You'll need to print the length of both lists.")
    _solution = CS("""print(len(my_numbers))
print(len(my_days))""")


class Exercise4(CodingProblem):
    _vars = ['linkedin_list']
    _hint = ("Use the sum() function to add all numbers in a list. "
             "For example: sum([1, 2, 3]) returns 6.")
    _solution = CS("sum(linkedin_list)")
    
    def check(self, linkedin_list):
        expected_sum = sum(linkedin_list)
        if expected_sum <= 70:
            raise AssertionError(f"The sum of linkedin_list is {expected_sum}. "
                               "Make sure your list contains values that sum to more than 70.")
        # If we get here, the sum is > 70 as expected


class Exercise5(CodingProblem):
    _vars = ['facebook_list', 'facebook_monday']
    _hint = ("List indexing starts at 0. So the first item in a list is at index [0]. "
             "To get Monday's value: variable_name = list_name[0]")
    _solution = CS("facebook_monday = facebook_list[0]")
    
    def check(self, facebook_list, facebook_monday):
        if facebook_monday != facebook_list[0]:
            raise AssertionError(f"Expected facebook_monday to be {facebook_list[0]} (the first item), "
                               f"but got {facebook_monday}. Remember that list indexing starts at 0.")


class Exercise6(CodingProblem):
    _vars = ['facebook_list', 'facebook']
    _hint = ("Use np.array() to convert a list to a NumPy array. "
             "For example: my_array = np.array(my_list)")
    _solution = CS("facebook = np.array(facebook_list)")
    
    def check(self, facebook_list, facebook):
        if not isinstance(facebook, np.ndarray):
            raise AssertionError("Expected facebook to be a NumPy array. "
                               "Use np.array() to convert the list.")
        
        expected_array = np.array(facebook_list)
        if not np.array_equal(facebook, expected_array):
            raise AssertionError(f"Expected facebook array to contain {facebook_list}, "
                               f"but got {facebook.tolist()}.")


class Exercise7(ThoughtExperiment):
    _hint = ("Use the == operator to check if two values are equal. "
             "For example: variable == 13 returns True if variable equals 13, False otherwise.")
    _solution = CS("views == 13")


class Exercise8(CodingProblem):
    _vars = ['num_views']
    _hint = ("Use an if statement to check if num_views is greater than 13. "
             "The syntax is: if (condition): followed by indented code.")
    _solution = CS("""if (num_views > 13):
    print('You are very popular!')""")
    
    def check(self, num_views):
        # This matches your original logic - just checking that num_views > 13
        if num_views <= 13:
            raise AssertionError("For this exercise, num_views should be greater than 13 "
                               "to demonstrate the if statement working.")


class Exercise9(ThoughtExperiment):
    _hint = ("Use a for loop with range(3) to iterate through numbers 0, 1, 2. "
             "Inside the loop, print each number squared using i**2. "
             "Remember to indent the code inside the loop!")
    _solution = CS("""for i in range(3):
    print(i**2)""")


class Exercise10(CodingProblem):
    _vars = ['facebook', 'linkedin']
    _hint = ("First add the two NumPy arrays together, then use np.mean() to calculate the average. "
             "NumPy arrays can be added with the + operator: array1 + array2")
    _solution = CS("np.mean(facebook + linkedin)")
    
    def check(self, facebook, linkedin):
        expected_avg = np.mean(facebook + linkedin)
        # Keep your original range check logic
        if not (expected_avg > 22 and expected_avg < 22.5):
            raise AssertionError(f"Expected the average to be between 22 and 22.5, "
                               f"but got {expected_avg:.2f}. Check your calculation.")


class Exercise11(ThoughtExperiment):
    _hint = ("Use a for loop with range(len(facebook_list)) to iterate through indices. "
             "Inside the loop, use an if-else statement to compare facebook_list[i] with linkedin_list[i]. "
             "Print 'Facebook wins.' or 'LinkedIn wins.' based on which value is higher.")
    _solution = CS("""for i in range(len(facebook_list)):
    if (facebook_list[i] > linkedin_list[i]):
        print('Facebook wins.')
    else:
        print('LinkedIn wins.')""")


class Exercise12(CodingProblem):
    _vars = ['freq', 'min_freq']
    _hint = ("Use np.min() to find the minimum value in a NumPy array. "
             "For example: np.min(my_array) returns the smallest value.")
    _solution = CS("min_freq = np.min(freq)")
    
    def check(self, freq, min_freq):
        expected_min = np.min(freq)
        if min_freq != expected_min:
            raise AssertionError(f"Expected min_freq to be {expected_min}, but got {min_freq}.")
        # Keep your original assertion that min should be 0
        if min_freq != 0:
            raise AssertionError(f"Expected the minimum frequency to be 0, but got {min_freq}.")


class Exercise13(ThoughtExperiment):
    _hint = ("Use boolean indexing to filter the array. When freq equals min_freq, "
             "you get a boolean array. Use this to index deathpenalty_array: "
             "deathpenalty_array[boolean_condition]")
    _solution = CS("deathpenalty_array[freq == min_freq]")


class Exercise14(ThoughtExperiment):
    _hint = ("Calculate the proportion by dividing the number of black defendants with death penalty "
             "by the total number of defendants with death penalty. "
             "Use: numerator / (numerator + denominator)")
    _solution = CS("n_black_deathpenalty / (n_black_deathpenalty + n_white_deathpenalty)")


class Exercise15(ThoughtExperiment):
    _hint = ("Create a boolean condition combining three criteria using & operator: "
             "(defendantrace == 1) & (victimrace == 0) & (deathpenalty == 1). "
             "Then sum the frequencies for these cases and divide by total white defendants with death penalty.")
    _solution = CS("""white_defendant_and_black_victim_and_deathpenalty = (defendantrace == 1) & (victimrace == 0) & (deathpenalty == 1)
np.sum(freq[white_defendant_and_black_victim_and_deathpenalty]) / n_white_deathpenalty""")


# Bind all exercises
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
