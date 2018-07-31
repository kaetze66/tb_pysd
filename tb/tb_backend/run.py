"""
run class is the main class to pass through in the tests
used for MP, but also single thread
links multiple forms of output and is the basis for the saving pipe

This should contain all the run information that Model.run requires, plus the output, plus the name
Some of the output might not be needed, hence they are all standardized in init

Version 0.2
Update 30.07.18/sk
"""

import numpy as np
from configparser import ConfigParser
import os


class Run:
    """
    run class handles a single run
    """

    def __init__(self, name, full_id, exo_names=None, params=None, return_columns=None, change='',
                 return_timestamps=None, reload=True):
        """
        there are a lot of inputs for this one so it might be better with *args

        :param name: str, name of the run
        :param full_id: str, group of full df that the run belongs to
        :param exo_names: list of parameters to be adjusted
        :param params: list of parameter values
        :param return_columns: list of columns to be returned
        :param change: change is used when only one parameter is changed from the base run
        :param return_timestamps: list of time stamps to be returned
        :param reload: Whether the model should be reloaded, generally should be True
        """
        self.name = name
        self.full_id = full_id
        if exo_names is not None:
            self.input_dict = dict(zip(exo_names, params))
        else:
            self.input_dict = {}
        self.return_columns = return_columns
        self.return_timestamps = return_timestamps
        self.initial_condition = 'Original'
        self.reload = reload
        self.change = change
        # initializing the variable type dict
        self.var_dict = {'only pos': [], 'only neg': [], 'pos and neg': []}
        self.cf = ConfigParser()
        # this needs to be checked because of os.getcwd() 30.07.18/sk
        # otherwise we need to bring the model path down
        self.cf.read(os.path.join(os.getcwd().replace('\\run_pipe',''), '_config', 'settings.ini'))
        # precision for rounding and finding negative flows and stocks
        # unlimited precision is sometimes generating negative values where there shouldn't be any
        # defining the precision, decimal points after the comma
        self.precision = self.cf['model'].getint('round_precision', fallback=8)
        # defining infinity to avoid graphing problems with matplotlib
        self.infinity = self.cf['model'].getfloat('infinity', fallback=1.00e+100)

    def add_run(self, run):
        """
        add run to the run object

        :param run: pd.Dataframe() with run results
        :return:
        """
        self.run = run

    def add_change(self, change):
        """
        Adding the change parameter to the run object

        Deprecated 27.07.18/sk
        Not used 27.07.18/sk

        :param change:
        :return:
        """
        self.change = change

    def add_output_names(self, names):
        """
        adding the return columns

        Deprecated 27.07.18/sk
        Not used 27.07.18/sk

        :param names:
        :return:
        """
        self.return_columns = names

    def create_norm(self):
        """
        Create the normalized run with respect to values of t=0

        :return:
        """
        self.norm = (self.run - self.run.iloc[0]) / self.run.iloc[0]

    def create_sens(self, base):
        """
        Create the sensitivity run with respect to another run

        :param base: run to get sensitivity from, usually the base run
        :return:
        """
        self.sens = (self.run - base).divide(base.abs(), fill_value=1)

    def calc_fit(self, base):
        """
        calculates the difference of two runs on one or multiple variables,
        important is that base and run are the same dimensions

        :param base: df with x variables, usually endogenous, compared to
        :return: float, result of the sum of sums of squared errors
        """
        fit = (base - self.run) ** 2
        self.fit = fit.sum(axis=0).sum(axis=0)

    def var_types(self):
        """
        Gather the variable types:
        - always positive
        - always negative
        - neither

        returns var dict filled

        :return:
        """
        var_lst = list(self.run)
        # sometimes pulse functions are not recognized as float64, so here we make sure it does
        self.run = self.run.astype(np.float64)
        for var in var_lst:
            # if something is off beyond 8 decimal places, then we shouldn't care
            # positive and negative infinity also is not necessary here (causes error caught in error file)
            if (self.run[var].replace([np.inf, -np.inf], np.nan).dropna().round(self.precision) >= 0).all():
                self.var_dict['only pos'].append(var)
            elif (self.run[var].replace([np.inf, -np.inf], np.nan).dropna().round(self.precision) <= 0).all():
                self.var_dict['only neg'].append(var)
            else:
                self.var_dict['pos and neg'].append(var)

    def chk_run(self):
        """
        Checks the run for very large values and replaces them with np.nan
        Also replaces np.inf with np.nan to make run checking easier
        for the full df a check exists to see if there is a np.nan at t=0 and if there is, it doesn't graph it
        If it were np.inf, it would graph it even though we don't want that there

        :return:
        """
        # making sure the values in the runs can be graphed
        # technically we should set them to np.inf, but since we're going to put np.inf to np.nan, so we save some work
        mask = self.run > self.infinity
        self.run[mask] = np.nan
        mask = self.run < -1 * self.infinity
        self.run[mask] = np.nan
        self.run.replace([-np.inf, np.inf], np.nan, inplace=True)
        # checking at t=0 if any np.nan exist and eliminates the run if so
        if self.run.iloc[0].isnull().values.any():
            return False
        else:
            return True

    def treat_run(self, base):
        """
        summary method to treat a run

        only called when needed 27.07.18/sk

        :param base: pd.Dataframe(), results of the base run
        :return:
        """
        self.create_norm()
        self.create_sens(base)
        self.var_types()
