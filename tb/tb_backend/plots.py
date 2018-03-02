"""
This file includes all required plotting operations in the testing battery

- plots
- pairplots
- heatmaps
- clustermaps

Version 0.1
Update 02.03.18/sk
"""

import matplotlib.pyplot as plt
from tb.tb_backend import fileops as fops
import seaborn as sns
import pandas as pd
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (20,12)
plt.tight_layout()


def init(base_unit):
    """
    Defines the globals for plots
    """
    global time_unit
    global ylabel_dict
    time_unit = base_unit
    # no dict value for endo_sens as there it's the unit of the endogenous variable
    ylabel_dict = {'run': 'various units', 'norm': '% change compared to var at t=0',
                   'exo_sens': '% change compared to base run'}


def create_plot(run,ptype,name,test,unit=None):
    """
    creates and saves the plots
    :param run: the df with the data from the run
    :param ptype: can be run, norm, exo_sens, endo_sens
    :param name: name of the plot, used for title
    :param test: source of the plot, e.g. equi, sens
    :param unit: unit (optional), for plots with the same units on ylabel
    :return: plot to be saved (in fileops)
    """
    run.plot()
    ymin,ymax = plt.ylim()
    plt.xlabel(time_unit)
    if unit is not None:
        plt.ylabel(unit)
    else:
        plt.ylabel(ylabel_dict[ptype])
    plt.title(name)
    # for exo_sens and norm graphs percentages higher than 1000% are not that interesting,
    # if larger limits are used, then the more relevant smaller changes do not show
    if ptype in ['exo_sens', 'norm']:
        if ymin < -10:
            plt.ylim(ymin=-10)
        if ymax > 10:
            plt.ylim(ymax=10)

    savename = '%s_%s' % (ptype,name)
    return fops.save_plots(savename,test)

# creates the pairplot
def create_pairgrid(run,name,test):
    """
    This creates the pairplot for sens, there are three parts (diag, upper, lower)
    upper and lower represent the same pairings and thus could display different graphs
    :param run: df with data to plot
    :param name: string, name of the plot
    :param test: string, name of the originating test
    :return: plot to be saved (in fileops)
    """
    g = sns.PairGrid(run)
    try:
        g.map_diag(plt.hist)
    except:
        pass
    g.map_upper(plt.scatter)
    #for lower kde plots are an option but take time to create, currently scatter is used (even though redundant)
    g.map_lower(plt.scatter)
    #g.map_lower(sns.kdeplot)
    savename = 'PP_%s' % name
    return fops.save_plots(savename,test)

def create_endo_sens_plots(endo_names,doc,full_df,sp,test):
    """
    create the sensitivity graphs with all sensitivity runs for each endogenous variable

    :param endo_names: list, names of the endogenous variables
    :param doc: df with the documentation information
    :param full_df: df, complete with all runs to be plotted
    :param sp: sensitivity percentage, used for naming the plots
    :param test: originating test for saving
    :return: plots to be created (with plot function)
    """

    type_dict = {'sens': 'endo_sens','equi': 'equi'}
    for i, var in enumerate(endo_names):
        # sensitivity percentage is only relevant for sensitivity graphs
        if test == 'sens':
            name = '%s_%s' % (var, sp)
        else:
            name = var
        unit = doc.loc[doc['Real Name'] == var]['Unit'].values
        endo_sens = full_df.loc(axis=1)[:, var]
        endo_sens.columns = endo_sens.columns.droplevel(1)
        create_plot(endo_sens, type_dict[test], name, test, unit)
        plt.close('all')

def create_heatmap(df,sp,nmb,test):
    """
    function to create the heatmap
    heatmap slices the full df at specific times that are predefined and shows every exo-endo pairing at this time
    the number of heatmaps determines the time intervals (i.e. the total time horizon is split between the
    number of heatmaps)

    also creates the clustermaps with the same principle

    :param df: df, full_df with all the runs
    :param sp: sensitivity percentage, used for naming
    :param nmb: integer, number of heatmaps to be created
    :param test: originating test for saving
    :return: saved heatmaps and saved clustermaps
    """
    #might need a better way to define default f_time, also user_input could overwrite
    #this is done on index level to avoid problems with dates in the index
    nmb_heatmaps = nmb
    interval = round((len(df.index)-1)/nmb_heatmaps,0)

    # creating the heatmaps and clustermaps, the full df needs to be a bit adjusted for this operation
    # creates slices at specific times in the simulation
    for i in range(nmb_heatmaps):
        # j saves the index for the time step, this is important if dates are used in the models
        j = int(interval*(i+1))
        f_time = df.index[j]
        sens_title = 'sensitivity in percent at t=%s' % f_time
        # creates the time slice and uses it as sum_df
        sum_df = df.loc(axis=0)[f_time]
        sum_df = pd.DataFrame(sum_df).reset_index()
        # the new column headers are introduced to make sure the pivot works
        sum_df.columns = ['run',sens_title, '']
        # creating a pivot table with the runs on the left and endogenous variables on the top
        sum_df = sum_df.pivot_table(sum_df,index='run',columns=sens_title)
        # dropping the highest level of the multi index which is now empty (was third column in sum_df)
        sum_df.columns = sum_df.columns.droplevel(0)
        # calculate sensitivities by dividing every sensitivity run by the base run
        sum_df = (sum_df-sum_df.loc['base'])/sum_df.loc['base']
        # sum_df could be saved here, but information is not that relevant
        # if to be saved, there needs to be numbering based on i, as they are otherwise overwritten
        # then rows with empty or faulty values are eliminated to create the maps, this means that potentially
        # runs are missing in the graphs
        sum_df = sum_df.dropna(axis=1)
        # then we transpose the df, to have the exogenous variables on the x-axis (cause)
        # and endogenous on the y-axis (effect)
        sum_df = sum_df.transpose()
        # create the maps and we're done
        sns.heatmap(sum_df,cmap='coolwarm',annot=True)
        fops.save_plots('heatmap_%s_%s.png' % (sens_title,sp),test)
        sns.clustermap(sum_df,cmap='coolwarm')
        fops.save_plots('clustermap_%s_%s.png' % (sens_title, sp), test)
        plt.close('all')