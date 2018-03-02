"""
this code sets models to equilbrium and evaluates the equilibria

version 0.1
02.03.18/sk
"""

import pysd
import os
import pandas as pd
import numpy as np
import scipy.optimize as opt
from timeit import default_timer as timer
from tb.tb_backend import fileops as fops
from tb.tb_backend import utils
from tb.tb_backend import plots


def config(equilibrium_percentage=0.1, equi_method=1):
    """
    setting the equilibrium config parameters
    with equi method 2, larger equilibrium percentages might be a good idea, since bound increase is not
    dependent on iter_count

    equilibrium percentage needs to be passed through here because globals are set after config


    :param equilibrium_percentage: float, increase of bounds for each iteration in
    :param equi_method: method of how incremental equilibrium finding is handled
    :return: equilibrium percentage, initiation of model and error counts, time lists and name for error file
    """
    sp = equilibrium_percentage
    # method 1 increases bounds for exogenous variables by 10% in the direction of where the solver is hitting
    # the bounds on the basis of the initial value in the base run
    # method 2 increases bounds by +- 10% based on the solver result from last iteration
    equi_method = equi_method
    equi_model_count = 0
    equi_time_lst = []
    # output file for errors when the model is run
    equi_error_file = 'equilibrium_error_file.txt'
    equi_error_cnt = 0
    return sp, equi_method, equi_model_count, equi_time_lst, equi_error_file, equi_error_cnt

def init(test_dir,equilibrium_percentage):
    """
    setting the globals for equilibrium tests
    max_equi is currently hard coded, if there's benefit to it, it might be moved to settings

    :param test_dir: string of base folder
    :param equilibrium_percentage: float, percentage increase for bounds in incremental equilibrium
    :return:
    """
    global folder
    folder = test_dir
    global sp
    sp = equilibrium_percentage
    # max equi is used if equilbrium is not searched for incrementally
    # max_equi are for testing when there are almost no iteration, performance comparison
    global max_equi
    max_equi = (0,10)
    # source is needed for the error file
    global source
    source = os.path.basename(__file__)
    global test
    test = 'equi'

def equilibrium(model_file,equi_method,incremental=True):
    """
    The equilibrium function per se, reads the model in and searches for an equilbrium based on passed settings
    returns the equilibrium run and equilbrium documentation as well as found errors

    :param model_file: string, name of .py file to be worked on, file extension included
    :param equi_method: int, either 1 or 2, defines the method used for equilibrium search
    :param incremental: Boolean, to set search to incremental (method 1 or 2) or not (using max_equi)
    :return: list with errors found at runtime
    """
    output_name = model_file.split('.')[0]
    # output folder is equi to differentiate from sens
    fops.output_folder(output_name,test)

    doc = fops.read_doc_file(output_name)
    # this ensures that switches are not changed
    doc = utils.ID_switches(doc)
    const, builtin, stocks, endo, _, _ = utils.get_type_df(doc)
    flow_expr_lst = utils.op_flows(stocks)
    utils.read_base_unit(doc)
    model = pysd.load(os.path.join(folder, model_file))
    base_full = model.run()
    const, _, init_lst = utils.set_base_params(base_full, const, builtin)
    err_lst = []

    bound_lst = utils.create_init_bounds(init_lst,sp,max_equi,incremental)
    endo_names, stock_names, flow_names, exo_names = utils.create_name_lists(stocks, endo, const)

    full_df = pd.DataFrame()
    base = base_full[endo_names]
    full_df = utils.add_run(full_df, base, 'base')

    res_lst = []
    param_lst = np.array(init_lst)
    iter_cnt = 1

    # equilbrium function
    def equilibrium(param_lst,err_lst):
        res, err_lst = utils.run_with_tracking(model,dict(zip(exo_names,param_lst)),endo_names,err_lst,source)
        equi = utils.calc_equi(flow_expr_lst,res)
        return equi

    # optimizer
    res_eq = opt.minimize(equilibrium, param_lst, args=err_lst, bounds=bound_lst)

    #builds dict with result
    eq_run = utils.create_run_with_result(model,res_eq.x)

    # fit checks how far the equilibrium is from the base run, closer is better
    fit = utils.calc_fit(base,eq_run)
    res_lst.append((res_eq.fun, fit, res_eq.x))
    improv = 1

    # this manages the iteration for finding the equilibrium, is potentially an infinite loop,
    # improv makes sure that it rarely is even when no equilibrium can be found
    # this is the equilibrium function per se
    while res_eq.fun > 0.1 and improv > 0:
        iter_cnt += 1
        bound_lst, param_lst = utils.build_bounds(bound_lst, res_eq.x, param_lst, init_lst, iter_cnt, sp, equi_method)
        res_eq = opt.minimize(equilibrium, param_lst, args=err_lst, bounds=bound_lst)
        eq_run = utils.create_run_with_result(model, res_eq.x)
        fit = utils.calc_fit(base, eq_run)
        res_lst.append((res_eq.fun, fit, res_eq.x))
        #calculates the difference between the last two iterations to set to equilibrium,
        # is -2 and -1 because the index in the list is one behind the count
        improv = res_lst[iter_cnt-2][0] - res_lst[iter_cnt-1][0]
        #print('Iteration:', iter_cnt)

    fops.save_lst_csv(res_lst,'equi_sum_%s' % output_name,test)

    # checking if found equilibrium has set the flows to 0 (bad equilibrium)
    exo_r_dict = utils.check_equilibrium(eq_run,res_eq.fun,res_eq.x)

    full_df = utils.add_run(full_df,eq_run,'equilibrium run')
    # plots all variables to check the found equilibrium
    plots.create_endo_sens_plots(endo_names,doc,full_df,sp,test)
    fops.write_equi_to_doc(exo_r_dict,doc,'%s_doc' % output_name,'doc')

    ind_end = timer()

    if not err_lst:
        err_lst.append((source,'No Error','','','',''))

    return err_lst









