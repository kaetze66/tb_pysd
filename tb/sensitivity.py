"""
this code runs all sensitivity related tests

version 0.1
02.03.18/sk
"""

import pysd
import os
import pandas as pd
import matplotlib.pyplot as plt
from tb.tb_backend import fileops as fops
from tb.tb_backend import plots
from tb.tb_backend import utils

def init(test_dir,sp):
    global folder
    folder = test_dir
    global sp_lst
    sp_lst = [1 * sp, -1 * sp]
    global source
    source = os.path.basename(__file__)
    global test
    test = 'sens'


def config(sensitivity_percentage=0.1,equi_mode=False):
    """
    setting sensitivity config parameters

    :param sensitivity_percentage: float, percentage of change for all exogenous variables for sensitivity runs
    :param equi_mode: Boolean, if True, equi values are used for initial guess if they exist, if False, base run values
    :return: sensitivity percentage, model and error counts, time list and error file name
    """
    sp = sensitivity_percentage
    equi_mode = equi_mode
    sens_model_count = 0
    sens_time_lst = []
    # output file for errors when the model is run
    sens_error_file = 'sensitivity_error_file.txt'
    sens_error_cnt = 0
    return sp,equi_mode,sens_model_count,sens_time_lst,sens_error_file,sens_error_cnt

def sensitivity(model_file,equi_mode):
    """
    This is the sensitivity test per se

    Produces all outputs with regard to sensitivity tests (e.g. run for positive and negative sensitivities)


    :param model_file: string, name of .py file to be worked on, file extension included
    :param equi_mode: Boolean, if true then equi params are used, if false, base params are used
    :return: list with errors at runtime
    """
    # Here all the information is read in and prepared
    # model is with extension
    output_name = model_file.split('.')[0]
    #this might be better in the battery itself, tbd
    fops.output_folder(output_name, test)
    doc = fops.read_doc_file(output_name)
    #only needs to be run once
    #needs to go to the testing battery eventually
    doc = utils.ID_switches(doc)
    const,builtin,stocks,endo,flows,switches = utils.get_type_df(doc)
    utils.read_base_unit(doc)
    endo_names, stock_names, flow_names, exo_names = utils.create_name_lists(stocks, endo, const)
    err_lst = []

    # Model is loaded and base parameters are set
    model = pysd.load(os.path.join(folder, model_file))
    base_full = model.run()
    const, _, base_params = utils.set_base_params(base_full,const,builtin,equi_mode)

    # create two loops, once for positive sensitivity percentage, once for negative
    for i, sp in enumerate(sp_lst):
        full_df = pd.DataFrame()
        base = base_full[endo_names]
        full_df = utils.add_run(full_df,base,'base')

        # runs sensitivity for all constants (exogenous variables)
        for i, row in const.iterrows():
            name = '%s_%s' % (row['Real Name'],sp)

            #working params ensures that only one variable is changed
            w_params = base_params.copy()
            w_params.iloc[i] *= (1+sp)

            run, err_lst = utils.run_with_tracking(model,dict(zip(exo_names,w_params)),endo_names,err_lst,source,
                                                       '%s=%s' % (row['Real Name'],w_params.iloc[i]))

            # exo sens calculates the percentage change of all endogenous variables compared to the base run
            exo_sens = (run - base) / base
            # exo_sens = exo_sens.replace(np.nan, 1)
            # norm calculates the percentage change of all endogenous variables compared to the value at t=0
            norm = (run - run.loc[0]) / run.loc[0]

            # saving all plots
            plots.create_plot(run,'run',name,test)
            # for normalized graphs only plots that don't have 0 as initial variable are shown
            plots.create_plot(norm,'norm',name,test)
            plots.create_plot(exo_sens,'exo_sens',name,test)
            plots.create_pairgrid(exo_sens,name,test)
            #pairgrid_save(exo_sens,output_folder,name)
            plt.close('all')

            # saving the csv files mostly for testing, not sure if useful for modeler
            fops.save_csv(run,'run_%s' % name,test)
            fops.save_csv(norm, 'norm_%s' % name, test)
            fops.save_csv(exo_sens, 'exo_sens_%s' % name, test)

            # adding the run to the full df
            full_df = utils.add_run(full_df,run,name)

        # this is just to avoid errors with types
        full_df = full_df.astype(float)
        fops.save_csv(full_df, 'full_df_%s' %sp, test)

        plots.create_endo_sens_plots(endo_names,doc,full_df,sp,test)

        plots.create_heatmap(full_df,sp,4,test)

    if not err_lst:
        err_lst.append((source, 'No Error', '', '', '', ''))

    return err_lst
