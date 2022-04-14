# -*- coding: utf-8 -*-
"""
Some decorators useful for data analysis functions
"""

def multi_input(f):
    """Allow a function to also be applied to each element in a dictionary"""
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
    """Allow running a function multiple times returning the average output"""
    def wrapper(average=1, *args, **kwargs):
        data_av = f(*args, **kwargs)
        for i in range(average-1):
            data_av += f(*args, **kwargs)
        data_av /= average
        return data_av
    return wrapper