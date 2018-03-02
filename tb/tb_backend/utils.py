"""
This file includes all required data operations in the testing battery

- reading and adapting read in information (dfs, doc, etc.)
-

Version 0.1
Update 02.03.18/sk
"""

from tb.tb_backend import plots
import pandas as pd
import numpy as np
import traceback
import re

def ID_switches(doc_df):
    """
    this is necessary as for descriptives the switches are counted as constants
    only needs to be run once

    Switches will be treated only as binary switches with 0 and 1 as permissible settings

    Switch settings will be treated as different models, i.e. tests are run with setting 0,
    then with setting 1 (not implemented yet)
    this might not be useful though, for now the modeler has to set the switches that they want and the testing battery
    doesn't touch them

    Switches are defined based on the name as there is no other property to identify them with
    :param doc_df: Dataframe with doc information
    :return: doc_df: Dataframe with doc information with type 'switch' added
    """

    for i, row in doc_df.iterrows():
        if 'switch' in row['Py Name'].split('_') and row['type'] == 'constant':
            row['type'] = 'switch'
    return doc_df

def get_type_df(doc_df):
    """
    not all types are used at all times

    unused types can be blanked out with underscores

    type dfs need to be used for input in tests as they contain the pynames and values

    :param doc_df: Dataframe with doc information
    :return: dataframes for different types of variables
    """
    const = doc_df.loc[doc_df['type'] == 'constant'].reset_index()
    builtin = doc_df.loc[doc_df['type'] == 'builtin'].reset_index()
    stocks = doc_df.loc[doc_df['type'] == 'stock'].reset_index()
    endo = doc_df.loc[~doc_df['type'].isin(['constant', 'builtin', 'stock', 'table function', 'subscript list',
                                            'subscripted constant','switch'])].reset_index()
    flows = doc_df.loc[doc_df['type'] == 'flow'].reset_index()
    switches = doc_df.loc[doc_df['type'] == 'switch'].reset_index()
    return const,builtin,stocks,endo,flows,switches

def read_base_unit(doc_df):
    """
    reads in the base unit for graphs, used on the x-axis
    automatically is read in for the init function of plots, where it will be used

    :param doc_df: Dataframe with doc information
    :return: base_unit
    """

    return plots.init(doc_df['Base Unit'][0])

def set_base_params(base,const,builtin,equi_mode=False):
    """
    This function reads the base parameters in. If equilibrium has been run and has a good equlibrium,
    then the equilibrium values are used if equimethod is set to True. Else, reads in the values from
    the base run and sets those as starting values

    mode is only important for sensitivity, for others it's not relevant,
    setting it to false equals to starting with base values

    :param base: Dataframe with base run (the model as is read in) output
    :param const: dataframe with const information
    :param builtin: dataframe with Vensim builtins
    :param equi_mode: boolean that is only necessary for sensitivity tests
    :return: updated const and builtin df, as well as params as an pd.series
    """
    if equi_mode and const['equi'][0] not in ['BE', 'NE']:
        base_params = const['equi']
    else:
        const,builtin = read_exo_base(base,const,builtin)
        base_params = const['value']
    return const, builtin, base_params



def read_exo_base(base,const,builtin):
    """
    reads in the values of all constants and builtins (vensim builtins),
    to have a starting point for test analysis

    :param base: Dataframe with base run (the model as is read in) output
    :param const: dataframe with const information
    :param builtin: dataframe with Vensim builtins
    :return: updated const and builtin dataframes
    """
    for i, row in const.iterrows():
        const.loc[i,'value'] = base[row['Real Name']].iloc[0]
    for i, row in builtin.iterrows():
        builtin.loc[i,'value'] = base[row['Real Name']].iloc[0]
    return const,builtin


def create_name_lists(stocks,endo,const):
    """
    creates list with names of endogenous variables to have output list when the model is run
    while params for input dicts in runs work wit py names and values, for output real names are used,
    thus these are stored in different lists

    the name lists are stored as globals and used
    in the other utils directly, without passing it back and forth from the tests

    all are real name based except exo, where the list contains pynames

    :param stocks: df of stocks
    :param endo: df of endogenous variables (flows are included in this one)
    :param const: df of constants
    :return: lists (endo, stock, flow) for outputs, pd.series for exo
    """
    global endo_names, flow_names, stock_names, exo_names
    endo_names = endo['Real Name'].tolist()
    flow_names = endo[endo['type'] == 'flow']['Real Name'].tolist()
    stock_names = stocks['Real Name'].tolist()
    endo_names.extend(stock_names)
    # exo_names needs to keep the order of the const DataFrame, list or pd.series are possible
    # currently pd.series is chosen but list would be prettier with the other lists (where order is not relevant)
    exo_names = const['Py Name']
    return endo_names, stock_names, flow_names, exo_names



def add_run(full_df,new_df,name):
    """
    adds run to full.df, which is the csv file that contains all the endogenous variables with all the runs
    full df is also used then to create the heatmaps

    :param full_df: df with all the runs up collected (only endogenous variables)
    :param new_df: df with the new run (only endogenous variables)
    :param name: name of the new run to be added, is added as level 1 in the multi index of the full df
    :return: full_df with added new run
    """
    new_df = pd.concat([new_df], axis=1, keys=[name])
    full_df = pd.concat([full_df,new_df],axis=1)
    return full_df


def run_with_tracking(model, param_dict, return_list, error_list, source, change=''):
    """
    tracks errors at runtime and reports those errors
    errors are logged in the error file with the information structure:
    source, error_type, description, location, parameter, change to base
    source is the name of the test and comes from the test.init functions
    error_type is created by this routine, currently includes
        division by 0, floating point, negative stock, negative flow
    description is explanatory text (division by 0, floating point) or
        the timesteps at which the error occures (negative stock, negative flow)
    location is the variable at which the error occured
    parameter: is the dict with all parameter settings
    change to base: is used when only one variable is changed compared to base (for easier viewing)
    :param model: model object returned from the load function from pysd
    :param param_dict: dictionary with the settings for the exogenous variables
    :param return_list: list with names of variables the pysd.run function should return
    :param error_list: list with the tracked errors, will translate to error_file.csv at end of test
    :param source: string, filename of the test
    :param change: string, slice of the param_dict that was changed
    :return: run results and updated error list
    """
    def clean_trace(trace_str):
        # cleans the traceback string to the relevant line and cleans it, returns location of runtime error
        trace_str = trace_str.splitlines()[-2]
        trace_str = trace_str.replace('return','').replace('()','').replace(' ','').strip()
        return trace_str
    def clean_type(type_str):
        # cleans the type of error if it's based on a warning at runtime and returns the error type
        return str(type_str).split('\'')[1]
    def clean_desc(desc_str):
        # cleans the error description and returns the description when the error is a warning at runtime
        desc_str = str(desc_str)
        if desc_str == 'invalid value encountered in double_scalars':
            desc_str = 'Invalid Value as Result'
        elif desc_str == 'divide by zero encountered in double_scalars':
            desc_str = 'Division by Zero'
        return desc_str
    def neg_flow_check(run):
        # checks negative flow values unless the flow name contains the word 'net'
        flow_df = run[flow_names]
        for column in flow_df:
            if 'net' not in column.lower().split(' '):
                # reports time stamps at which this error occurs (index is timestamps for run)
                neg_lst = flow_df.index[flow_df[column] < 0].tolist()
                if neg_lst:
                    error_list.append((source,'Negative Flow', neg_lst, column, param_dict, change))
    def neg_stock_check(run):
        # checks negative flow values, there is no keyword for stocks that could also go negative, should be declared
        stock_df = run[stock_names]
        for column in stock_df:
            # reports time stamps at which this error occurs (index is timestamps for run)
            neg_lst = stock_df.index[stock_df[column] < 0].tolist()
            if neg_lst:
                error_list.append((source, 'Negative Stock', neg_lst, column, param_dict, change))
    # creating the empty data frame to store the run in
    run = pd.DataFrame()
    # the model is run with np.errorstate in order to change the warnings to exceptions and being able to capture them
    # this will break the run, thus the timestamp for exception errors cannot be retrieved
    with np.errstate(divide='raise',invalid='raise'):
        try:
            run = model.run(params=param_dict, return_columns=return_list)
        except Exception as e:
            error_list.append((source, clean_type(type(e)), clean_desc(e), clean_trace(traceback.format_exc()),
                            param_dict, change))
    # if an exception occurs above, the run remains empty as execution is halted, thus the run has to be executed
    # again, but outside of np.errstate to just have warnings, that way the output is always a completed run
    if run.empty:
        run = model.run(params=param_dict, return_columns=return_list)
    # then the negative flows and stock checks are run on the completed run
    neg_flow_check(run)
    neg_stock_check(run)
    return run, error_list


def op_flows(stocks):
    """
    extracts the flows for the equilibrium calculation
    flow expressions are stored with the associated stocks in the stocks dataframe,
    thus having operations and names in the expression

    :param stocks: df with the stocks information
    :return: list of lists with the flow expressions split up
    """
    expr_lst = []
    for i, row in stocks.iterrows():
        flow_expr = row['flow expr']
        # split on + and - (only operations allowed in stocks) and keep the operator in the list
        flow_expr = re.split(r'([+-])', flow_expr)
        # strip all expressions to make sure that there are no errors due to spaces still in the thing
        flow_expr = [s.strip() for s in flow_expr]
        expr_lst.append(flow_expr)
    return expr_lst

# function to calculate the equilibrium in equilibrium.py
def calc_equi(expr_lst,res):
    """
    calculates the equilbrium result
    first calculates the sum of the flows for each stock
    then calculates the sum of absolute sums
    this sum needs to be 0 for an equilibrium to exist

    :param expr_lst: list of list of the flow expressions
    :param res: OptimizeResult object from scipy.optimize.minimize
    :return: sum of absolute sums
    """
    tot_res = 0
    # iterates through level 1 of list of lists
    for expr in expr_lst:
        st_res = 0
        # iterates through level 2 of the list of lists
        for i, el in enumerate(expr):
            # empty string needs to be in selection because if the first flow is negative, it will add an
            # empty string element to the expr (which is a list of strings)
            if el not in ['+','-', '']:
                out = res[el]
                if expr[i-1] == '-':
                    out = -out
                # calculates the stock result
                st_res += out
        tot_res += sum(abs(st_res))
    return tot_res

def create_init_bounds(params,sp,max_equi,incremental=True):
    """
    incremental=True indicates that equilibria closer to the base run are searched,
    is more time intensive than incremental = false
    creates the initial bounds for the equilibrium function

    even with incremental = False equilibria can still found incrementally as even with very large max_equi bounds,
    there is the possibility that incrementally the bounds are increased, but it's unlikelier

    :param params: dict with initial settings
    :param sp: float with equilibrium percentage change for iterations
    :param max_equi: tuple with minimum and maximum settings for non-incremental equilibrium finding
    :param incremental: boolean to set whether or not equilibria are found incrementally
    :return: list of tuples with the bounds for each exogenous variable in the model
    """

    bound_lst = []
    if incremental:
        for i, value in params.iteritems():
            # if values are 0 at t0 they need to be manually set to an arbitrary bounds, otherwise they won't change
            # not sure how to set them effectively
            if value == 0:
                bound_lst.append((0,1))
            else:
                bounds = (value * (1 - sp), value * (1 + sp))
                bound_lst.append(bounds)
    else:
        for i, value in params.iteritems():
            # if values are 0 at t0 they need to be manually set to an arbitrary bounds, otherwise they won't change
            # not sure how to set them effectively
            if value == 0:
                bound_lst.append(max_equi)
            else:
                bounds = (value * max_equi[0], value * max_equi[1])
                bound_lst.append(bounds)
    return bound_lst


def build_bounds(bounds,result,params,init,iter_cnt,sp,equi_method):
    """
    updates the bounds for each iteration of the solver
    method one increases the bounds based on the initial parameter value from the base run
    method two increases the bounds based on the result of the equilibrium function

    :param bounds: list of tuples containing the bounds from the previous iteration
    :param result: list (res.x) of the result of the previous iteration
    :param params: list, initial guess from previous iteration
    :param init: list, initial parameter settings, used for method one
    :param iter_cnt: integer, count of how many iterations it has gone through
    :param sp: float, defines the step each iteration increases the bounds
    :param equi_method: integer, indicates method
    :return: updated bounds list, parameters for next iteration
    """
    if equi_method == 1:
        for i, var in enumerate(result):
            lb,ub = bounds[i]
            #again we need to check if the initial is 0, then changed it to the result for bounds calculation
            if init.loc[i] == 0:
                # if initial parameter is zero, parameter is handled as if method 2 even though method 1 is selected
                # except that the applied space here is dependent on iter_cnt
                value = var
            else:
                value = init.loc[i]
            if lb == var:
                lb = value * (1-iter_cnt*sp)
            elif ub == var:
                ub = value * (1+iter_cnt*sp)
            bounds[i] = (lb,ub)
        params = result
    elif equi_method == 2:
        for i, var in enumerate(result):
            lb = var * (1-sp)
            ub = var * (1+sp)
            bounds[i] = (lb,ub)
        params = result
    return bounds, params

def calc_fit(base,run):
    """
    calculates the difference of two runs on one or multiple variables,
    important is that base and run are the same dimensions
    :param base: df with x variables, usually endogenous, compared to
    :param run: df with x variables, usually endogenous, compared from
    :return: float, result of the sum of sums of squared differences
    """
    fit = (base - run) ** 2
    fit = fit.sum(axis=0).sum(axis=0)
    return fit

def check_equilibrium(run,result,values):
    """
    this function checks the result of the equilibrium function and adjusts it if not all conditions for
    a good equilibrium are met
    if the sum for the equilibrium is 0, but the sum of all flows is 0, then an equilibrium was found, but it is just
    by setting all parameters to 0, thus making it impossible to use for other tests, thus the values are
    changed to BE, bad equilibrium

    if the result of the equilibrium function is larger than 0.1, then no equilibrium could be found, thus changing
    the values to NE, no equilibrium

    it is possible that no equilibrium is found because the while loop of the equilibrium function exists due to
    improvement being 0 even tough an equilibrium might be possible, but I don't know how to fix that
    :param run: df with the run results (endogenous variables)
    :param result: float, input is the res.fun value of the optimization
    :param values: the last parameter values found in the equilibrium function
    :return: the updated dictionary with equi values (or NE, BE)
    """
    equi_dict = dict(zip(exo_names,values))
    if run[flow_names].iloc[0].sum(axis=0) == 0:
        for key, val in equi_dict.items():
            equi_dict[key] = 'BE'
    if result > 0.1:
        for key, val in equi_dict.items():
            equi_dict[key] = 'NE'
    return equi_dict

def create_run_with_result(model,result):
    """
    creates a run with the results of some function, does not need to pass exo names because exo names are
    global in here
    :param model: model object created by the pysd.load function
    :param result: list or series with parameter settings
    :return: df with the resulting run (endogenous variables)
    """
    res_dict = dict(zip(exo_names, result))
    run = model.run(params=res_dict, return_columns=endo_names)
    return run
