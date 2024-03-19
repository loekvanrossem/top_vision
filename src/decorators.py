"""
Some decorators useful for data analysis functions
"""


def multi_input(f):
    """
    Allow a function to also be applied to each element in a dictionary.

    This decorator allows a function to be applied to each element in a dictionary.
    If the input data is a dictionary, the function f is applied to each element in
    the dictionary separately. The decorator then returns a dictionary containing the
    results. If the input is not a dictionary, the function is applied directly.
    This decorator is useful when dealing with multiple datasets stored in a dictionary.
    """

    def wrapper(data, *args, **kwargs):
        if type(data) is dict:
            output_data = {}
            for name in data:
                output_data[name] = f(data[name], *args, **kwargs)
            if all(x is None for x in output_data.values()):
                return
            else:
                return output_data
        else:
            return f(data, *args, **kwargs)

    return wrapper


def av_output(f):
    """
    Allow running a function multiple times returning the average output.

    This decorator allows running a function multiple times and returning the average
    output. The decorator takes an argument average, which specifies the number of
    times the function should be executed. The function f is executed average times,
    and the results are summed. Finally, the summed results are divided by average to
    compute the average output. This decorator is helpful when you want to obtain a
    smoother or more stable output by averaging over multiple runs.
    """

    def wrapper(average=1, *args, **kwargs):
        data_av = f(*args, **kwargs)
        for i in range(average - 1):
            data_av += f(*args, **kwargs)
        data_av /= average
        return data_av

    return wrapper
