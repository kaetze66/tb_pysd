"""
model class is the augmented model class that expands on the pysd model class with additional information

- allows attachment of a full_df with all the runs
- allows run with tracking for run time error tracking

Version 0.2
Update 30.07.18/sk
"""

import pandas as pd
from timeit import default_timer as timer
import numpy as np
import traceback
from pysd.py_backend.functions import Model, Time
from configparser import ConfigParser
import os


class AugModel(Model):
    """
    Augmodel is built on top of the pysd model class to add functionality
    """

    def __init__(self, model_path):
        """
        Careful with this init... model.reload() is going to initialize the model again, so stuff that needs to be kept
        beyond the reload cannot be in the init function

        :param model_path:
        """
        super(Model, self).__init__(model_path, None, None)
        self.time = Time()
        self.time.stage = 'Load'
        self.initialize()

    def set_tracking(self):
        """
        This needs to be outside of init because model reload starts resets the model but we would like to keep
        the error list and the full df dict

        essentially all the init that shouldn't be reloaded

        :return:
        """
        self.cf = ConfigParser()
        # if this is run with MP, then the current working directory is the run pipe
        # if not, then it is ok
        self.cf.read(os.path.join(os.getcwd().replace('\\run_pipe',''), '_config', 'settings.ini'))
        # precision for rounding and finding negative flows and stocks
        # unlimited precision is sometimes generating negative values where there shouldn't be any
        self.precision = self.cf['model'].getint('round_precision', fallback=8)
        self.max_ts = self.cf['model'].getint('max_ts', fallback=10)
        # list for tracking errors
        self.err_lst = []
        # full df dict initialized
        self.full_df_dict = {}

    def start_model(self):
        """
        Starting the timer for the model

        Deprecated 27.07.18/sk
        Not used 27.07.18/sk

        :return:
        """
        self.start = timer()

    def end_model(self):
        """
        Ending the timer for the model

        Deprecated 27.07.18/sk
        Not used 27.07.18/sk

        :return:
        """
        self.end = timer()
        self.elapsed = (self.end - self.start) / 60

    def create_full_df(self, base, group):
        """
        Creating the full_df by group with the base run as initial point

        :param base: base run from the model
        :param group: identifier for the full_df dict
        :return:
        """
        self.full_df_dict[group] = pd.DataFrame()
        self.add_run(base, 'base', group)

    def add_run(self, new_df, name, group):
        """
        adds run to full.df, which is the Dataframe that contains all the endogenous variables with all the runs
        full df is also used then to create the heatmaps

        :param group: identifier for the full_df dict
        :param new_df: df with the new run (only endogenous variables)
        :param name: name of the new run to be added, is added as level 1 in the multi index of the full df
        :return: full_df with added new run
        """
        new_df = pd.concat([new_df], axis=1, keys=[name])
        new_df = new_df.astype('float64', copy=False)
        self.full_df_dict[group] = pd.concat([self.full_df_dict[group], new_df], axis=1)

    @staticmethod
    def clean_trace(trace_str):
        """
        cleans the traceback string to the relevant line and cleans it, returns location of runtime error

        input is:
        Traceback (most recent call last):
          File "C:\winprog\Anaconda3\lib\site-packages\pysd\py_backend\functions.py", line 71, in cached
            assert cached.cache_t == func.__globals__['time']()
        AttributeError: 'function' object has no attribute 'cache_t'
        During handling of the above exception, another exception occurred:
        Traceback (most recent call last):
          File "C:\code\testingbattery\tb\tb_backend\model.py", line 224, in run_with_tracking
            run.reload)
          File "C:\winprog\Anaconda3\lib\site-packages\pysd\py_backend\functions.py", line 648, in run
            res = self._integrate(t_series, capture_elements, return_timestamps)
          File "C:\winprog\Anaconda3\lib\site-packages\pysd\py_backend\functions.py", line 747, in _integrate
            outputs.append({key: getattr(self.components, key)() for key in capture_elements})
          File "C:\winprog\Anaconda3\lib\site-packages\pysd\py_backend\functions.py", line 747, in <dictcomp>
            outputs.append({key: getattr(self.components, key)() for key in capture_elements})
          File "C:\winprog\Anaconda3\lib\site-packages\pysd\py_backend\functions.py", line 75, in cached
            cached.cache_val = func(*args)
          File "C:\code\testingbattery\test180711\workforce_treated.py", line 293, in pressure_to_hire
            return (task_backlog() - target_backlog()) / target_backlog()
        ZeroDivisionError: float division by zero

        Output is:
        pressure_to_hire

        :param trace_str: error trace
        :return:
        """
        lines = trace_str.splitlines()
        i = -2
        while True:
            # checks until it finds the last file, and saves i
            if lines[i].replace('\t', '').replace(' ', '').startswith('File'):
                break
            else:
                i -= 1
        # returns the last element of the line before the return
        return lines[i].rsplit(' ', 1)[-1].strip()

    @staticmethod
    def clean_type(type_str):
        """
        cleans the type of error if it's based on a warning at runtime and returns the error type

        input is:
        <class 'ZeroDivisionError'>

        output is:
        ZeroDivisionError

        :param type_str:
        :return:
        """
        return str(type_str).split('\'')[1]

    @staticmethod
    def clean_desc(desc_str):
        """
        cleans the error description and returns the description when the error is a warning at runtime

        input:
        'invalid value encountered in double_scalars'

        output:
        'Invalid Value as Result'

        :param desc_str: cleans up the description for the error file
        :return:
        """

        desc_str = str(desc_str)
        if desc_str == 'invalid value encountered in double_scalars':
            desc_str = 'Invalid Value as Result'
        elif desc_str == 'divide by zero encountered in double_scalars':
            desc_str = 'Division by Zero'
        return desc_str

    def neg_flow_check(self, run, param_dict, flow_names, test_name, change=''):
        """
        checks negative flow values unless the flow name contains the word 'net'


        :param run: pd.Dataframe(), result of the model execution
        :param param_dict: dict, parameter settings for the run
        :param flow_names: list, flows in the model
        :param test_name: str, name of the test
        :param change: str, if only one variable is changed, the variable is reported here
        :return:
        """

        errors = []
        flow_df = run[flow_names]
        # iterating over each flow in the df
        for column in flow_df:
            # ignores the flows that have net in the name
            if 'net' not in column.lower().split(' '):
                # reports time stamps at which this error occurs (index is timestamps for run)
                # we don't have to account for the np.nan here (unlike in the run comparison)
                # because np.nan inequalities are always FALSE
                neg_lst = flow_df.index[flow_df[column].round(self.precision) < 0].tolist()
                # only the first 10 time steps are reported, to keep the error file clean
                neg_lst = neg_lst[:self.max_ts]
                if neg_lst:
                    errors.append((test_name, 'Negative Flow', neg_lst, column, param_dict, change))
        return errors

    def neg_stock_check(self, run, param_dict, stock_names, test_name, change=''):
        """
        checks negative stock values, there is no keyword for stocks that could also go negative, should be declared

        :param run: pd.Dataframe(), run results
        :param param_dict: dict, parameter settings for the run
        :param stock_names: list, name of stocks in the model
        :param test_name: str, test the run originates from
        :param change: str, if only one variable is changed, it is reported here
        :return:
        """
        errors = []
        stock_df = run[stock_names]
        # iterating over each stock
        for column in stock_df:
            # reports time stamps at which this error occurs (index is timestamps for run)
            # we don't have to account for the np.nan here (unlike in the run comparison)
            # because np.nan inequalities are always FALSE
            neg_lst = stock_df.index[stock_df[column].round(self.precision) < 0].tolist()
            # only the first 10 time steps are reported, to keep the error file clean
            neg_lst = neg_lst[:self.max_ts]
            if neg_lst:
                errors.append((test_name, 'Negative Stock', neg_lst, column, param_dict, change))
        return errors

    def run_with_tracking(self, args):
        """
        args = run,flow_names,stock_names,test_name

        returns a tuple with three elements
        name, res, errors

        tracks errors at runtime and reports those errors

        errors are logged in the error file with the information structure:
        source, error_type, description, location, parameter, change to base
        source is the name of the test and comes from the test.init functions
        error_type is created by this routine, currently includes
            division by 0, floating point, negative stock, negative flow
        description is explanatory text (division by 0, floating point) or
            the timesteps at which the error occurs (negative stock, negative flow)
        location is the variable at which the error occured
        parameter: is the dict with all parameter settings
        change to base: is used when only one variable is changed compared to base (for easier viewing)

        :param args: tuple with run info, flow names, stock names and test name
        :return: run results and updated error list
        """
        run, flow_names, stock_names, test_name = args
        # creating the empty data frame to store the run in
        res = pd.DataFrame()
        errors = []
        # the model is run with np.errorstate in order to change the warnings to exceptions
        # and being able to capture them
        # this will break the run, thus the timestamp for exception errors cannot be retrieved
        with np.errstate(divide='raise', invalid='raise'):
            try:
                res = self.run(run.input_dict, run.return_columns, run.return_timestamps, run.initial_condition,
                               run.reload)
            except Exception as e:
                errors.append(
                    (test_name, self.clean_type(type(e)), self.clean_desc(e), self.clean_trace(traceback.format_exc()),
                     run.input_dict, run.change))
        # if an exception occurs above, the run remains empty as execution is halted, thus the run has to be executed
        # again, but outside of np.errstate to just have warnings, that way the output is always a completed run
        # if a invalid value is hard coded (e.g. in the knockout analysis), then it still raises an error, so the try
        # block takes care of that
        if res.empty:
            try:
                res = self.run(run.input_dict, run.return_columns, run.return_timestamps, run.initial_condition,
                               run.reload)
                # then the negative flows and stock checks are run on the completed run
            except Exception:
                pass
        # the testing needs to be checked at the end
        if not res.empty:
            nf_errors = self.neg_flow_check(res, run.input_dict, flow_names, test_name, run.change)
            ns_errors = self.neg_stock_check(res, run.input_dict, stock_names, test_name, run.change)
            errors.extend(nf_errors)
            errors.extend(ns_errors)
        # this has to be fixed, the returning element should be only the run object with the information attached
        # also then the run check could be done here instead of in collect results (where there's no MP) 250718/sk
        return run.name, res, errors

    def add_errors(self, errors):
        """
        Adds errors to the error list

        Deprecated 27.07.18/sk
        Not used 27.07.18/sk

        :param errors:
        :return:
        """
        self.err_lst.extend(errors)

    def check_err_list(self, source):
        """
        Error list check to make sure that at least one entry happens to avoid problems when converting to DF


        :param source: test where the check originates from
        :return:
        """
        try:
            if not self.err_lst:
                self.err_lst.append((source, 'No Error', '', '', '', ''))
        except AttributeError:
            print('run time error')
