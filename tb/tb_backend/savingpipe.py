"""
saving pipe includes all saving and plotting operation from runs
saves plots, heatmaps, csv to the respective folders
attached to a saving operation

Version 0.2
Update 30.07.18/sk
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from cycler import cycler
import math
from graphviz import Digraph
import pickle
from configparser import ConfigParser


class Plass:
    """
    the Pipeline for plotting and saving (Plass) handles all the plotting and saving
    """

    def __init__(self, test):
        self.test = test.test
        self.folder_dict = test.folder_dict
        self.base_unit = test.base_unit
        self.endo_names = test.endo_names
        self.stock_names = test.stock_names
        self.constants = test.const
        self.tables = test.tables
        self.switches = test.switches
        self.doc = test.doc
        # no dict value for endo_run as there it's the unit of the endogenous variable
        self.ylabel_dict = {'run': 'various units', 'norm': '% change compared to var at t=0',
                            'exo_sens': '% change compared to base run', 'endo_sens': '% change compared to base run'}
        # config file is defined and read in here
        self.cf = ConfigParser()
        # config folder doesn't change, so it's put here to it's on one line
        self.cf.read(os.path.join(os.path.split(test.folder)[0], '_config', 'tb_config.ini'))
        self.plot_limit = self.cf['saving pipe settings'].getint('plot_limit', fallback=20)
        self.testing_mode = self.cf['testing'].getboolean('testing_mode', fallback=False)
        plt.rcParams['figure.figsize'] = (self.cf['saving pipe settings'].getint('fig_width', fallback=20),
                                          self.cf['saving pipe settings'].getint('fig_height', fallback=12))
        plt.rcParams['font.size'] = 18
        self.model_width = self.cf['saving pipe settings'].getint('model_width', fallback=20)
        # we need to add the testing mode here to remove the cluster maps 02.07.18/sk
        # the full df dict should not be necessary anymore 03.06.18/sk
        # self.full_df_dict = test.model.full_df_dict
        # config folder doesn't change, so it's put here to it's on one line
        self.cf.read(os.path.join(os.path.split(test.folder)[0], '_config', 'settings.ini'))
        self.base_lw = self.cf['savingpipe'].getint('base_lw', fallback=4)
        self.model_format = self.cf['savingpipe'].get('model_format', fallback='svg')
        self.model_fs = self.cf['savingpipe'].get('model_fs', fallback='8')
        self.bound_inner = self.cf['savingpipe'].getfloat('bound_inner', fallback=0.1)
        self.bound_middle = self.cf['savingpipe'].getfloat('bound_middle', fallback=0.25)
        self.bound_outer = self.cf['savingpipe'].getfloat('bound_outer', fallback=0.4)
        self.anim_int = self.cf['savingpipe'].getint('anim_int', fallback=10000)

        self.run_lst = test.run_lst

        self.saveper = test.base_builtin.loc[2]
        self.initial = test.base_builtin.loc[1]
        self.edge_lst = test.edge_lst
        self.node_lst = test.node_lst
        sns.set_style('whitegrid')
        plt.tight_layout()
        plt.rcParams['axes.prop_cycle'] = (cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                                                         '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                                                         '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                                                         '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']) +
                                           cycler(linestyle=['-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
                                                             '--', '--', '--', '--', '--', '--', '--', '--', '--', '--',
                                                             ':', ':', ':', ':', ':', ':', ':', ':', ':', ':',
                                                             '-.', '-.', '-.', '-.', '-.', '-.', '-.', '-.', '-.',
                                                             '-.']))

    def create_sens_plot(self, full, unit, name, ptype):
        """
        Creating plots for the output

        :param full: pd.Dataframe(), with all the results
        :param unit: str, unit for endogenous graphs
        :param name: str, name of the graph
        :param ptype: str, type of graph to be produced
        :return:
        """
        # is limited to 20 for endo and exo plots, setting is in the pipe init
        its = 0
        if len(full.columns) > self.plot_limit:
            its = int(math.ceil(len(full.columns) / self.plot_limit))
            for it in range(its):
                fig, axes = plt.subplots()
                # base is always in the graph
                out_lst = ['base', ]
                out_lst.extend(full.columns[it * (self.plot_limit - 1) + 1:(it + 1) * (self.plot_limit - 1) + 1])
                plot_df = full.loc(axis=1)[out_lst]
                # this is just for testing and can be deleted when lize's model passes through
                # plot_df.to_csv(os.path.join(self.folder_dict[self.test],'%s_%s_%s.csv' % (ptype, name, it)))
                plot_df.plot(ax=axes)
                full.loc(axis=1)['base'].plot(color='#1f77b4', lw=self.base_lw, ax=axes, linestyle='-')
                plt.xlabel(self.base_unit)
                # creating the different y labels depending on type
                if ptype == 'endo_sens':
                    plt.ylabel(self.ylabel_dict[ptype])
                else:
                    plt.ylabel(unit)
                plt.title(name)
                savename = '%s_%s_%s' % (ptype, name, it)
                self.save_plots(savename)
                plt.close('all')
        else:
            fig, axes = plt.subplots()
            full.plot(ax=axes)
            full.loc(axis=1)['base'].plot(color='#1f77b4', lw=self.base_lw, ax=axes, linestyle='-')
            plt.xlabel(self.base_unit)
            if ptype == 'endo_sens':
                plt.ylabel(self.ylabel_dict[ptype])
            else:
                plt.ylabel(unit)
            plt.title(name)
            savename = '%s_%s' % (ptype, name)
            self.save_plots(savename)
            plt.close('all')
        # we have to pickle this because with MP the passing of arguments is faulty
        pickle_out = open(os.path.join(self.folder_dict[self.test], 'endo_its.pickle'), 'wb')
        pickle.dump(its, pickle_out)
        pickle_out.close()

    def create_model(self, name, full, mtype):
        """
        they are coming in here per full df, so no need to iterate over the full df dict
        full df dict is still needed for identifying multiple models of the same type

        I'm really not happy how complicated the structure is in here 03.06.18/sk

        :param name: str, key in the full df dict
        :param full: pandas df, the full df for identifying when there are too many runs
        :param mtype: str, type of output for the model graphs
        :return: model representation of the test
        """

        key = name
        group = []
        group_dict = {}
        # format for the model represenation, is the best I have found so far
        model_format = self.model_format
        # renaming for sensitivity models, other models stay the same
        if self.test == 'sens':
            # this has to be like that because 10% (sp) could be changed and the model should still recognize that
            if float(name) > 0:
                name = '%s_positive_sensitivity' % mtype
                iconpath = os.path.join(self.folder_dict['graphviz'], 'up.png')
            elif float(name) < 0:
                name = '%s_negative_sensitivity' % mtype
                iconpath = os.path.join(self.folder_dict['graphviz'], 'down.png')
            else:
                # this shouldn't be necessary, but to make sure it's complete
                iconpath = os.path.join(self.folder_dict['graphviz'], 'rand.png')
        elif self.test == 'ext':
            # for extreme condition tests, the values could change as well but 0 is probably going to stick
            # tables get the random dot
            if 'mult' in name:
                if name.replace('mult', '') == '0':
                    iconpath = os.path.join(self.folder_dict['graphviz'], 'down.png')
                else:
                    iconpath = os.path.join(self.folder_dict['graphviz'], 'up.png')
            else:
                iconpath = os.path.join(self.folder_dict['graphviz'], 'rand.png')
        elif self.test == 'ko':
            iconpath = os.path.join(self.folder_dict['graphviz'], 'stop.png')
        else:
            # equi passes here
            # tstep, hori pass here (although they don't need that)
            # swit passes here
            # mc passes here
            iconpath = os.path.join(self.folder_dict['graphviz'], 'rand.png')
        const_lst = self.constants['Real Name'].tolist()
        # only in the sensitivity we can use the number of constants to calculate the number of runs
        # I'm not happy with how the full df are currently. It should be one full df, one model, however that is not
        # possible because they are different 03.06.18/sk
        # the problem is a bit that there are no common ways of setting the limit for graphs when there are more
        # than 20 runs per endo variable
        # MC and equi doesn't have that problem because it's always only one graph
        # extreme has full dfs for 0, 10 (multipliers) and one per table, which is inconsistent (tables should
        # probably be together)
        # iterations can be calculated based on full df columns,
        # marking of variables is different between the tests though
        runs = len(full.columns) / len(self.endo_names)
        # mc has more runs but they are plotted on one graph
        if runs > self.plot_limit and self.test != 'mc':
            its = int(math.ceil(runs / self.plot_limit))
        else:
            its = 0
        # for sensitivity and extreme condition tests, the constant list is the basis for indicating changes
        if self.test == 'sens':
            if its != 0:
                for it in range(its):
                    group = const_lst[it * (self.plot_limit - 1):(it + 1) * (self.plot_limit - 1)]
                    group_dict[it] = group
            else:
                group = const_lst
        elif self.test == 'ext':
            # ext has two different types of graphs, multipliers and table graphs
            if 'mult' in name:
                if its != 0:
                    for it in range(its):
                        group = const_lst[it * (self.plot_limit - 1):(it + 1) * (self.plot_limit - 1)]
                        group_dict[it] = group
                else:
                    group = const_lst
            else:
                table_id = int(name.replace('table', ''))
                # if it's a single variable, it needs to be converted to a list to get the marking consistent
                group = [self.tables.iloc[table_id]['Real Name']]
        elif self.test in ['hori', 'tstep']:
            group = []
        elif self.test == 'swit':
            if its != 0:
                for it in range(its):
                    group = self.switches['Real Name'].tolist()
                    group_dict[it] = group
            else:
                group = self.switches['Real Name'].tolist()
        # ko has one full df per variable because of visibility issues
        # mc is the same, but many runs
        elif self.test in ['mc', 'ko']:
            # if it's a single variable, it needs to be converted to a list to get the marking consistent
            group = [name]
        else:
            # equi passes through here
            group = const_lst
        if its == 0:
            dot = Digraph(name=name, format=model_format,
                          node_attr={'shape': 'box', 'color': 'white', 'fontsize': self.model_fs})
            # need to find a way to determine optimal size for used screen
            dot.attr(size='%s,100' % self.model_width)
            for node in self.node_lst:
                # all constants and table functions are preset with an image
                if node['image'] != '':
                    # only variables in the group are highlighted
                    if node['label'] in group:
                        dot.node(str(node['ID']), label=node['label'],
                                 image=iconpath,
                                 URL=node['URL'],
                                 fixedsize=node['fixedsize'], width=node['width'], height=node['height'],
                                 labelloc=node['labelloc'])
                    else:
                        dot.node(str(node['ID']), label=node['label'],
                                 image=node['image'],
                                 URL=node['URL'],
                                 fixedsize=node['fixedsize'], width=node['width'], height=node['height'],
                                 labelloc=node['labelloc'])
                else:
                    # stocks are different because they have their frame added again
                    # currently there is no test that manipulates stocks, so here we can just pass them to the graph
                    if node['label'] in self.stock_names:
                        target = '%s_%s_%s.png' % (mtype, node['label'], key)
                        savepath = os.path.join(self.folder_dict[self.test], target)
                        urlpath = 'file:./%s' % target
                        dot.node(str(node['ID']), label=node['label'],
                                 image=savepath,
                                 URL=urlpath,
                                 fixedsize=node['fixedsize'], width=node['width'], height=node['height'],
                                 labelloc=node['labelloc'], color='black')
                    else:
                        # this is necessary for knockout where changes are not on exogenous variables but endogenous
                        if node['label'] in group:
                            dot.node(str(node['ID']), label=node['label'],
                                     image=iconpath,
                                     URL=node['URL'],
                                     fixedsize=node['fixedsize'], width=node['width'], height=node['height'],
                                     labelloc=node['labelloc'])
                        else:
                            target = '%s_%s_%s.png' % (mtype, node['label'], key)
                            savepath = os.path.join(self.folder_dict[self.test], target)
                            urlpath = 'file:./%s' % target
                            dot.node(str(node['ID']), label=node['label'],
                                     image=savepath,
                                     URL=urlpath,
                                     fixedsize=node['fixedsize'], width=node['width'], height=node['height'],
                                     labelloc=node['labelloc'])
            for edge in self.edge_lst:
                dot.edge(str(edge[0]), str(edge[1]))
            dot.render(os.path.join(self.folder_dict[self.test], '%s.gv' % name), view=False, cleanup=True)
        else:
            for it, group in group_dict.items():
                # it name preserves the original name and adds the iteration
                itname = '%s_%s' % (name, it)
                # this is to preserve the original node list and have the neutral sign for all items not included
                # in the group
                itnode_lst = self.node_lst
                dot = Digraph(name=itname, format=model_format,
                              node_attr={'shape': 'box', 'color': 'white', 'fontsize': self.model_fs})
                for node in itnode_lst:
                    if node['image'] != '':
                        # only variables in the group are highlighted
                        if node['label'] in group:
                            dot.node(str(node['ID']), label=node['label'],
                                     image=iconpath,
                                     URL=node['URL'],
                                     fixedsize=node['fixedsize'], width=node['width'], height=node['height'],
                                     labelloc=node['labelloc'])
                        else:
                            dot.node(str(node['ID']), label=node['label'],
                                     image=node['image'],
                                     URL=node['URL'],
                                     fixedsize=node['fixedsize'], width=node['width'], height=node['height'],
                                     labelloc=node['labelloc'])
                    else:
                        # stocks are different because they have their frame readded
                        if node['label'] in self.stock_names:
                            target = '%s_%s_%s_%s.png' % (mtype, node['label'], key, it)
                            savepath = os.path.join(self.folder_dict[self.test], target)
                            urlpath = 'file:./%s' % target
                            dot.node(str(node['ID']), label=node['label'],
                                     image=savepath,
                                     URL=urlpath,
                                     fixedsize=node['fixedsize'], width=node['width'], height=node['height'],
                                     labelloc=node['labelloc'], color='black')
                        else:
                            target = '%s_%s_%s_%s.png' % (mtype, node['label'], key, it)
                            savepath = os.path.join(self.folder_dict[self.test], target)
                            urlpath = 'file:./%s' % target
                            dot.node(str(node['ID']), label=node['label'],
                                     image=savepath,
                                     URL=urlpath,
                                     fixedsize=node['fixedsize'], width=node['width'], height=node['height'],
                                     labelloc=node['labelloc'])
                for edge in self.edge_lst:
                    dot.edge(str(edge[0]), str(edge[1]))
                itname = itname.replace('"', '').replace('/', '').replace('*', '')
                dot.render(os.path.join(self.folder_dict[self.test], '%s.gv' % itname), view=False, cleanup=True)

    def create_mc_plot(self, full, unit, name, ptype):
        """
        creates the MC graph with the confidence bounds

        :param full: full df with the informatino
        :param unit: unit for the y axis
        :param name: name of the graph
        :param ptype: plot type, i.e. MC
        """
        full.iloc(axis=1)[1:-2].plot(color='black', legend=False, linestyle='-')
        full.loc(axis=1)['base'].plot(color='red', lw=self.base_lw, linestyle='-', legend=True, label='base')
        full.loc(axis=1)['floor'].plot(color='yellow', lw=self.base_lw, linestyle='-', legend=True, label='floor')
        full.loc(axis=1)['ceiling'].plot(color='orange', lw=self.base_lw, linestyle='-', legend=True, label='ceiling')
        # the quantiles would ideally be renamed 30.07.18/sk
        q10 = full.quantile(0.5 - self.bound_outer, axis=1)
        q25 = full.quantile(0.5 - self.bound_middle, axis=1)
        q40 = full.quantile(0.5 - self.bound_inner, axis=1)
        q60 = full.quantile(0.5 + self.bound_inner, axis=1)
        q75 = full.quantile(0.5 + self.bound_middle, axis=1)
        q90 = full.quantile(0.5 + self.bound_outer, axis=1)
        plt.fill_between(full.index, q10, q90, alpha=1, color='#549bf1')
        plt.fill_between(full.index, q25, q75, alpha=1, color='#3329eb')
        plt.fill_between(full.index, q40, q60, alpha=1, color='#00ff04')
        plt.xlabel(self.base_unit)
        plt.ylabel(unit)
        plt.title(name)
        savename = '%s_%s' % (ptype, name)
        self.save_plots(savename)
        plt.close('all')

    def create_timeline(self, df, name):
        """
        Creating the time line for the equilibrium tests

        :param df:
        :param name:
        :return:
        """
        df = df.transpose()
        sns.heatmap(df, cmap='coolwarm', vmin=0, vmax=1, cbar=False)
        self.save_plots('timeline_%s' % name)

    def create_heatmap(self, key, full_df, nmb):
        """
        function to create the heatmap
        heatmap slices the full df at specific times that are predefined and shows every exo-endo pairing at this time
        the number of heatmaps determines the time intervals (i.e. the total time horizon is split between the
        number of heatmaps)

        also creates the clustermaps with the same principle

        :param key: str, key for the full df that is currently worked
        :param full_df: full df for the type
        :param nmb: integer, number of heatmaps to be created
        :return: saved heatmaps and saved clustermaps
        """

        # might need a better way to define default f_time, also user_input could overwrite
        # this is done on index level to avoid problems with dates in the index
        f_time_lst = []
        nmb_heatmaps = nmb
        interval = round((len(full_df.index) - 1) / nmb_heatmaps, 0)

        # creating the heatmaps and clustermaps, the full df needs to be a bit adjusted for this operation
        # creates slices at specific times in the simulation
        for i in range(nmb_heatmaps):
            # j saves the index for the time step, this is important if dates are used in the models
            j = int(interval * (i + 1))
            f_time = full_df.index[j]
            f_time_lst.append(f_time)
            sens_title = 'sensitivity in percent at t=%s' % f_time
            # creates the time slice and uses it as sum_df
            sum_df = full_df.loc(axis=0)[f_time]
            sum_df = pd.DataFrame(sum_df).reset_index()
            # the new column headers are introduced to make sure the pivot works
            sum_df.columns = ['run', sens_title, '']
            # creating a pivot table with the runs on the left and endogenous variables on the top
            sum_df = sum_df.pivot_table(sum_df, index='run', columns=sens_title)
            # dropping the highest level of the multi index which is now empty (was third column in sum_df)
            # droplevel works on multiindex
            sum_df.columns = sum_df.columns.droplevel(0)
            # calculate sensitivities by dividing every sensitivity run by the base run
            sum_df = (sum_df - sum_df.loc['base']) / sum_df.loc['base']
            # sum_df could be saved here, but information is not that relevant
            # if to be saved, there needs to be numbering based on i, as they are otherwise overwritten

            # then we transpose the df, to have the exogenous variables on the x-axis (cause)
            # and endogenous on the y-axis (effect)
            sum_df = sum_df.transpose()
            # create the maps and we're done
            sns.heatmap(sum_df, cmap='coolwarm', annot=True, vmin=-1, vmax=1)
            self.save_plots('heatmap_%s_%s' % (sens_title, key))
            # then rows with empty or faulty values are eliminated to create the maps, this means that potentially
            # runs are missing in the graphs
            # the dropna is ok here because clustermaps are currently not used
            # this whole thing should be only availabe in testing mode
            if self.testing_mode:
                sum_df = sum_df.dropna(axis=0)
                sns.clustermap(sum_df, cmap='coolwarm')
                self.save_plots('clustermap_%s_%s' % (sens_title, key))
            plt.close('all')
        # we have to pickle this because with MP the passing of arguments is faulty
        pickle_out = open(os.path.join(self.folder_dict[self.test], 'intervals.pickle'), 'wb')
        pickle.dump(f_time_lst, pickle_out)
        pickle_out.close()

    def create_anim_heatmap(self, key, full_df):
        """
        Creating the animated heatmap
        this might be a bit problematic becaues it needs additional setup
        only called in testing mode

        :param key: str, name of the full df currently worked
        :param full_df: full df for the type
        :return:
        """

        def gen_heatmap(time=self.initial):
            """
            This produces the heatmaps that are then added to the animation

            :param time:
            :return:
            """
            # maybe the nan values should be blacked or something to check for serious issues
            # currently are reported as 0
            if time == self.initial:
                color_bar = True
            else:
                color_bar = False
            sens_title = 'sensitivity in percent at t=%s' % time
            sum_df = full_df.loc(axis=0)[time]
            sum_df = pd.DataFrame(sum_df).reset_index()
            sum_df.columns = ['run', sens_title, '']
            sum_df = sum_df.pivot_table(sum_df, index='run', columns=sens_title)
            # droplevel works on multiindex
            sum_df.columns = sum_df.columns.droplevel(0)
            sum_df = (sum_df - sum_df.loc['base']) / sum_df.loc['base']
            sum_df = sum_df.transpose()
            plt.title(sens_title)
            plt.tight_layout()
            return sns.heatmap(sum_df, cmap='coolwarm', cbar=color_bar, vmax=1, vmin=-1)

        fig = plt.figure()
        interval = int(1 / self.saveper)
        frame_lst = full_df.index[interval::interval]
        anim = FuncAnimation(fig, gen_heatmap, frame_lst, init_func=gen_heatmap, repeat=False, interval=self.anim_int)
        mywriter = animation.FFMpegWriter(fps=8)
        anim.save(os.path.join(self.folder_dict[self.test], 'anim_heatmap_%s.mp4' % key), writer=mywriter)
        plt.close('all')

    def create_pairgrid(self, run, name):
        """
        This creates the pairplot for sens, there are three parts (diag, upper, lower)
        upper and lower represent the same pairings and thus could display different graphs

        Deprecated 27.07.18/sk
        Not in use 27.07.18/sk
        might still be useful at some point


        :param run: df with data to plot
        :param name: string, name of the plot
        :return: plot to be saved (in fileops)
        """
        g = sns.PairGrid(run)
        try:
            g.map_diag(plt.hist)
        except:
            pass
        g.map_upper(plt.scatter)
        # for lower kde plots are an option but take time to create, currently scatter is used (even though redundant)
        g.map_lower(plt.scatter)
        # g.map_lower(sns.kdeplot)
        savename = 'PP_%s' % name
        self.save_plots(savename)
        plt.close('all')

    def create_plot(self, run, ptype, name):
        """
        # only exogenous ones are to be done
        :param run: the df with the data from the run
        :param ptype: can be run, norm, exo_sens, endo_sens
        :param name: name of the plot, used for title
        :return: plot to be saved (in fileops)
        """
        its = 0
        if len(run.columns) > self.plot_limit:
            its = int(math.ceil(len(run.columns) / self.plot_limit))
            for it in range(its):
                out_lst = run.columns[it * self.plot_limit:(it + 1) * self.plot_limit]
                run.loc(axis=1)[out_lst].plot()
                plt.xlabel(self.base_unit)
                plt.ylabel(self.ylabel_dict[ptype])
                plt.title(name)
                ymin, ymax = plt.ylim()
                # for exo_sens and norm graphs percentages higher than 1000% are not that interesting,
                # if larger limits are used, then the more relevant smaller changes do not show
                if ptype in ['exo_sens', 'norm']:
                    if ymin < -1:
                        plt.ylim(ymin=-1)
                    if ymax > 1:
                        plt.ylim(ymax=1)

                savename = '%s_%s_%s' % (ptype, name, it)
                self.save_plots(savename)
                plt.close('all')
        else:
            run.plot()
            plt.xlabel(self.base_unit)
            plt.ylabel(self.ylabel_dict[ptype])
            plt.title(name)
            ymin, ymax = plt.ylim()
            # for exo_sens and norm graphs percentages higher than 1000% are not that interesting,
            # if larger limits are used, then the more relevant smaller changes do not show
            if ptype in ['exo_sens', 'norm']:
                if ymin < -1:
                    plt.ylim(ymin=-1)
                if ymax > 1:
                    plt.ylim(ymax=1)

            savename = '%s_%s' % (ptype, name)
            self.save_plots(savename)
            plt.close('all')
        # we have to pickle this because with MP the passing of arguments is faulty
        pickle_out = open(os.path.join(self.folder_dict[self.test], 'exo_its.pickle'), 'wb')
        pickle.dump(its, pickle_out)
        pickle_out.close()

    # same concept for plots as for runs
    # is a bit more complicated as there are regular plots, heatmaps, clustermaps and montecarlo plots
    # only makes sense if this is entered into a pipeline
    # maybe also objects for csv files to be saved, I dunno
    def save_plots(self, plot_name):
        """
        This saves the plots to their respective location

        :param plot_name:
        :return: saved .png
        """
        # the replace elements are just to make sure that there are no naming error issues
        name = plot_name.replace('"', '').replace('/', '').replace('*', '')
        return plt.savefig(os.path.join(self.folder_dict[self.test], '%s.png' % name), bbox_inches='tight')

    def save_csv(self, df, ptype, csv_name):
        """
        Saves data to .csv, is used for output files that will be rewritten for every iteration
        :param ptype:
        :param df: Dataframe with the data
        :param csv_name: string, name of the output file
        :return: saved .csv
        """
        # the replace elements are just to make sure that there are no naming error issues
        csv_name = str(csv_name)
        name = csv_name.replace('.py', '').replace('"', '').replace('/', '').replace('*', '')
        return df.to_csv(os.path.join(self.folder_dict[self.test], '%s_%s.csv' % (ptype, name)), index=True,
                         header=True)
