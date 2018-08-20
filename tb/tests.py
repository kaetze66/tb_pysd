"""
tests class collects all the methods that are test specific

they need to be single thread and MP compatible

test structure is always:

- self.initialize_test()
- self.prepare_test()
- res = self.run_test()
- self.collect_res(res)
- self.save_full_output()
- self.save_ind_output()
- self.save_output()
- self.write_report()
- self.end_test()

For non MP use, there are helper functions that take care of iterating over the items:

- self.initialize_test()
- self.prepare_test()
- res = self.run_test_helper()
- self.collect_res(res)
- self.save_full_output_helper()
- self.save_ind_output_helper()
- self.save_output()
- self.write_report()
- self.end_test()

todo:
- check all run with tracking for appropriate reload setting

Version 0.3
Update 30.07.18/sk
"""

import os
import pandas as pd
import numpy as np
import numpy.random as rand
from tb.basetest import Test
from tb.tb_backend.run import Run
from tb.tb_backend.savingpipe import Plass
import re
import scipy.optimize as opt
import itertools
from ast import literal_eval
from tb.tb_backend.report import Report
import pickle
from configparser import ConfigParser


class Sensitivity(Test):
    """

    usage:

    from tb.tests import Sensitivity
    from tb.tb_backend.savingpipe import Plass
    folder = r'C:\code\testingbattery\FOLDER'
    test = Sensitivity(folder,'MODELNAME.py',0.1)
    test.initialize_test()
    test.prepare_test()
    res = test.run_test_helper()
    test.collect_res(res)
    # add the saving pipe stuff
    """

    def __init__(self, folder, file_name, sensitivity_percentage):
        super(Sensitivity, self).__init__(folder, file_name, 'sensitivity')
        self.err_list = []
        self.MP = True
        self.sp = sensitivity_percentage
        self.class_name = 'Sensitivity'
        # this needs to be integrated into the test definition in the battery and the builder
        self.equimode = False
        self.cf = ConfigParser()
        # config folder doesn't change, so it's put here to it's on one line
        self.cf.read(os.path.join(os.path.split(folder)[0], '_config', 'tb_config.ini'))
        # this should go to saving pipe
        self.nmb_heatmaps = self.cf['saving pipe settings'].getint('nmb_heatmaps', fallback=4)

    def set_equimode(self, equimode=False):
        """
        Deprecated 27.07.18/sk
        Not in use 27.07.18/sk

        :param equimode:
        :return:
        """

        # this really needs to be tested
        # this should not be necessary anmyore 02.07.18/sk
        self.equimode = equimode

    def set_base_params(self):
        """
        Setting of base parameters for the base run

        :return:
        """
        if self.equimode:
            self.base_params = self.const['equi']
        else:
            self.base_params = self.const['value']
        self.base_builtin = self.builtin['value']

    def prepare_test(self):
        """
        Prepares the sensitivity runs and adds them to the run list

        :return:
        """
        # creates one run for positive and one for negative sensitivity
        sp_lst = [self.sp * 1, self.sp * -1]
        for sp in sp_lst:
            # positive and negative sensitivity get a full df each
            self.model.create_full_df(self.base.run, sp)
            for i, row in self.const.iterrows():
                name = '%s_%s' % (row['Real Name'], sp)
                w_params = self.base_params.copy()
                w_params.iloc[i] *= (1 + sp)
                # Run has inputs name,full_ID,exo_names=None,params=None,return_columns=None
                self.run_lst.append(Run(name, sp, self.exo_names, w_params, self.endo_names,
                                        '%s=%s' % (row['Real Name'], w_params.iloc[i])))

    def collect_res(self, res):
        """
        collects the results from the test execution and prepares them for further use

        sens has additional sensitivity calcs for each run


        :param res: result list from run execution
        """
        for i, run in enumerate(self.run_lst):
            err_lst = res[i][2]
            if not res[i][1].empty:
                # this should be eliminated in a revision, results should come in as a list of run objects 250718/sk
                run.run = res[i][1].astype('float64', copy=False)
                # run.chk_run() tests if there are np.nan in the first line,
                # which means the run couldn't be executed properly and shouldn't be added
                # those runs should technically not even show up (some do, some don't)
                # topic to discuss with PySD 180722/sk
                if run.chk_run():
                    self.model.add_run(run.run, run.name, run.full_id)
                    run.treat_run(self.base.run)
                else:
                    # we remove negative stock and flow errors here because those runs are not supposed
                    # to have run in the first place
                    # negative stock and flow errors in this case arise from np.inf in some variables
                    # caused by division by 0
                    # while they technically should be fine, it's just confusing for anyone to have an error
                    # other than the division by 0
                    err_lst = [x for x in res[i][2] if x[1] not in ['Negative Flow', 'Negative Stock']]
                # print is just for testing
                print(i)
            self.model.err_lst.extend(err_lst)
        # opening pipes all over the place might be not the best idea, one pipe for all saving might be better
        pipe = Plass(self)
        for key, full_df in self.model.full_df_dict.items():
            pipe.save_csv(full_df, 'full_df', key)

    def save_ind_output_mp(self, run):
        """

        :param run:
        """
        pipe = Plass(self)
        pipe.create_plot(run.run, 'run', run.name)
        pipe.create_plot(run.norm, 'norm', run.name)
        pipe.create_plot(run.sens, 'exo_sens', run.name)
        if self.testing_mode:
            pipe.save_csv(run.run, 'run', run.name)
            pipe.save_csv(run.norm, 'norm', run.name)
            pipe.save_csv(run.sens, 'exo_sens', run.name)

    def save_full_output_mp(self, key):
        """
        Overwrite because for sens we need endo run and endo sens graphs and models

        :param key: key for the full_df
        :return:
        """

        full_df = pd.read_csv(os.path.join(self.folder_dict[self.test], 'full_df_%s.csv' % key), index_col=0,
                              header=[0, 1], dtype=np.float64)

        self.iterate_endo_plots(key, full_df)
        pipe = Plass(self)
        # this shouldn't be necessary anymore 26.07.18/sk
        # full_df = full_df.astype(float)
        pipe.create_heatmap(key, full_df, self.nmb_heatmaps)
        if self.testing_mode:
            try:
                pipe.create_anim_heatmap(key, full_df)
            # define the exception thrown here
            except:
                pass
        # probably need two iterations, one for endo sens and one for endo run, exos are not handled in a model
        pipe.create_model(key, full_df, 'endo_run')
        pipe.create_model(key, full_df, 'endo_sens')
        # this should not be necessary anymore 30.07.18/sk
        #if self.full_df_output:
        #    pipe.save_csv(full_df, 'full_df', key)

    def save_endo_plots(self, endo_run, unit, name):
        """

        :param endo_run:
        :param unit:
        :param name:
        """
        # type name now includes the prefix, if necessary
        pipe = Plass(self)
        type_name = 'endo_run'
        pipe.create_sens_plot(endo_run, unit, name, type_name)
        if self.testing_mode:
            pipe.save_csv(endo_run, type_name, name)
        # this transpose shouldn't be necessary, but division by first column doesn't seem to work
        endo_run = endo_run.transpose()
        endo_sens = (endo_run - endo_run.iloc[0]) / endo_run.iloc[0]
        endo_sens = endo_sens.transpose()
        type_name = 'endo_sens'
        pipe.create_sens_plot(endo_sens, unit, name, type_name)
        if self.testing_mode:
            pipe.save_csv(endo_sens, type_name, name)

    def write_report(self):
        """

        Writing the report, inputs come from pickle files

        For sensitivity we need the intervals pickle (for the heatmaps) as well as the endo_its and exo_its
        for multiple graphs of the same variable

        """
        rep = Report(self.folder, self.file)
        const_lst = self.const['Real Name'].tolist()
        # we have to pickle this because with MP, the passing of arguments is faulty
        f_path = os.path.join(self.folder_dict[self.test], 'intervals.pickle')
        pickle_in = open(f_path, 'rb')
        intervals = pickle.load(pickle_in)
        pickle_in.close()
        os.remove(f_path)
        # endo its are the iterations for endogenous graphs
        f_path = os.path.join(self.folder_dict[self.test], 'endo_its.pickle')
        pickle_in = open(f_path, 'rb')
        endo_its = pickle.load(pickle_in)
        pickle_in.close()
        os.remove(f_path)
        # exo its are the iterations for exogenous graphs
        f_path = os.path.join(self.folder_dict[self.test], 'exo_its.pickle')
        pickle_in = open(f_path, 'rb')
        exo_its = pickle.load(pickle_in)
        pickle_in.close()
        os.remove(f_path)
        # report tuple includes section title, constant list, sensitivity percentage, intervals for the heatmap,
        # exogenous and endogenous iterations, link to test source folder
        rep_tpl = (self.class_name, const_lst, self.sp, intervals, exo_its, endo_its,
                   self.folder_dict[self.test].replace(self.folder_dict['source'], '').lstrip('\\'))
        rep.write_sens(rep_tpl)
        rep.save_report()


class MonteCarlo(Test):
    """

    Monte Carlo is a subclass of test and runs the MC testing

    """
    def __init__(self, folder, file_name, sensitivity_percentage, runs):
        super(MonteCarlo, self).__init__(folder, file_name, 'montecarlo')
        self.err_list = []
        self.MP = True
        self.sp = sensitivity_percentage
        self.nmb_runs = runs
        self.class_name = 'MonteCarlo'

    def prepare_test(self):
        """
        Prepares the runs and adds them to the run list
        Creates 100 random uniform runs for each parameter

        """
        for i, row in self.const.iterrows():
            self.model.create_full_df(self.base.run, row['Real Name'])
            if self.base_params.iloc[i] != 0:
                input_set = rand.uniform((1 - self.sp) * self.base_params.iloc[i],
                                         (1 + self.sp) * self.base_params.iloc[i],
                                         self.nmb_runs)
            else:
                input_set = np.full(1, 0)
            for j in np.nditer(input_set):
                name = '%s_%s' % (row['Real Name'], j)

                w_params = self.base_params.copy()
                w_params.iloc[i] = j
                # Run has inputs name,full_ID,exo_names=None,params=None,return_columns=None
                self.run_lst.append(Run(name, row['Real Name'], self.exo_names, w_params, self.endo_names,
                                        '%s=%s' % (row['Real Name'], w_params.iloc[i]), reload=True))
            w_params = self.base_params.copy()
            w_params.iloc[i] *= (1 - self.sp)
            self.run_lst.append(Run('floor', row['Real Name'], self.exo_names, w_params, self.endo_names,
                                    '%s=%s' % (row['Real Name'], w_params.iloc[i]), reload=True))
            w_params = self.base_params.copy()
            w_params.iloc[i] *= (1 + self.sp)
            self.run_lst.append(Run('ceiling', row['Real Name'], self.exo_names, w_params, self.endo_names,
                                    '%s=%s' % (row['Real Name'], w_params.iloc[i]), reload=True))

    def save_full_output_mp(self, key):
        """

        :param key:
        """
        full_df = pd.read_csv(os.path.join(self.folder_dict[self.test], 'full_df_%s.csv' % key), index_col=0,
                              header=[0, 1], dtype=np.float64)
        pipe = Plass(self)
        full_df = full_df.astype(float)
        self.iterate_endo_plots(key, full_df)
        if self.full_df_output:
            pipe.save_csv(full_df, 'full_df', key)
        pipe.create_model(key, full_df, self.test)

    def save_endo_plots(self, endo_run, unit, name):
        """

        :param endo_run:
        :param unit:
        :param name:
        """
        pipe = Plass(self)
        # type name now includes the prefix, if necessary
        type_name = self.test
        pipe.create_mc_plot(endo_run, unit, name, type_name)

    def write_report(self):
        """
        Writes the report for the MC test

        doesn't need any pickled information


        """
        rep = Report(self.folder, self.file)
        const_lst = self.const['Real Name'].tolist()
        # report tuple includes section title, constant list, MC percentage, link to test source
        rep_tpl = (self.class_name, const_lst, self.sp,
                   self.folder_dict[self.test].replace(self.folder_dict['source'], '').lstrip('\\'))
        rep.write_mc(rep_tpl)
        rep.save_report()


class Equilibrium(Test):
    """

    Saving of plots is generic (from Test class)
    """

    def __init__(self, folder, file_name, equi_method, increment_percentage, incremental=True):
        super(Equilibrium, self).__init__(folder, file_name, 'equilibrium')
        self.err_list = []
        self.MP = False
        self.sp = increment_percentage
        self.set_inc = incremental
        self.equi_method = equi_method
        self.class_name = 'Equilibrium'
        # sum df is summarizing the equi conditions found
        self.sum_df = None
        self.equi_set = {}
        self.equi_excl = []
        self.cf = ConfigParser()
        # config folder doesn't change, so it's put here to it's on one line
        self.cf.read(os.path.join(os.path.split(folder)[0], '_config', 'tb_config.ini'))
        self.equi_precision = self.cf['test parameters'].getfloat('equi_precision', fallback=0.01)
        self.cf.read(os.path.join(os.path.split(folder)[0], '_config', 'settings.ini'))
        self.equi_res = self.cf['tests'].getfloat('equi_res', fallback=0.1)
        self.equi_iter = self.cf['tests'].getfloat('equi_iter', fallback=0)
        self.equi_maxiter = self.cf['tests'].getint('equi_maxiter', fallback=20)

    def initialize_test(self, equimode=False):
        """

        :param equimode:
        """
        self.initialize_base()
        self.read_equi_file()
        self.op_flows()

    def read_equi_file(self):
        """
        Equi file is the file where the user inputs concerning the equilibrium test are stored

        """
        equi_file = '%s_equi.csv' % self.out_name
        equi_doc = pd.read_csv(os.path.join(self.folder_dict['doc'], equi_file), index_col=0)

        for i, row in equi_doc.iterrows():
            self.equi_set[row['Py Name']] = (row['fix value'], row['global minimum'], row['global maximum'])
            # if the value is fixed, its name is added to the excluded list
            if not np.isnan(row['fix value']):
                self.equi_excl.append(row['Py Name'])

    # equilbrium function
    def equilibrium(self, param_lst):
        """

        :param param_lst:
        :return:
        """
        name = ''
        run = Run(name, self.test, self.exo_names, param_lst, self.endo_names)
        args = run, self.flow_names, self.stock_names, self.test_name
        _, res, errors = self.model.run_with_tracking(args)
        equi = self.calc_equi(res)
        # runtime errors are tracked in the model class
        self.model.err_lst.extend(errors)
        return equi

    def collect_equi(self, name, equi_df, ts, index_lst):
        """
        recursively groups all equilibria conditions for the stocks and the model

        :param name: name of source, equi or base
        :param equi_df: dataframe of the run
        :param ts: timestep of the model
        :param index_lst: list with indices where there is an equilibrium condition
        :return:
        """
        cut_off = None
        ending_ts = None
        # while there are time steps in the index list, we continue
        if index_lst:
            initial_ts = index_lst[0]
            # if the length of the list is just 1 element, we have to capture that otherwise we get a max recursion
            # depth error
            if len(index_lst) > 1:
                # here we search forward until we find a time step that is after a gap
                for i, index in enumerate(index_lst):
                    if i > 0:
                        if index_lst[i] - index_lst[i - 1] != ts:
                            ending_ts = index_lst[i - 1]
                            cut_off = i
                            break
                if ending_ts is None:
                    ending_ts = index_lst[-1]
                    index_lst = []
            else:
                ending_ts = initial_ts
                index_lst = []
            # here we prepare the next iteration of the index list, if it's empty, it will stay empty
            index_lst = index_lst[cut_off:]
            st_lst = equi_df[self.stock_names].loc[initial_ts].tolist()
            sum_dict = {'name': name, 'start': initial_ts, 'end': ending_ts}
            for i, value in enumerate(self.stock_names):
                sum_dict[value] = st_lst[i]
            self.sum_df = self.sum_df.append(sum_dict, ignore_index=True)
            return self.collect_equi(name, equi_df, ts, index_lst)
        else:
            return

    def src_equi(self, run, name):
        """

        :param run: dataframe to search equilibrium conditions in
        :param name: name of the run
        """
        # we start off by adding the stocks to the equi_df because we need them for the initial conditions
        equi_df = run[self.stock_names]
        # equi_df = pd.concat([equi_df,run[self.flow_names]],axis=1)
        # iterates through the first level of the list with the flow expressions
        # tot res will be a pd.Series
        tot_res = 0
        for i, expr in enumerate(self.flow_expr_lst):
            st_res = 0
            st_name = self.stock_names[i]
            # iterates through the different elements in the flow expressions
            for j, el in enumerate(expr):
                if el not in ['+', '-', '']:
                    if expr[j - 1] == '-':
                        st_res -= run[el]
                    else:
                        st_res += run[el]
            st_res.name = 'sum_%s' % st_name
            # the threshold for equilibria is set at 0.01
            st_res[st_res.abs() < self.equi_precision] = 0
            equi_df = pd.concat([equi_df, st_res], axis=1)
            tot_res += st_res.abs()
        tot_res.name = 'model'
        equi_df = pd.concat([equi_df, tot_res], axis=1)
        self.save_csv('equi_df_%s' % name, equi_df, self.test)
        index_lst = equi_df.loc[equi_df['model'] == 0].index.tolist()
        ts = self.builtin['value'][2]
        self.collect_equi(name, equi_df, ts, index_lst)
        if name == 'base':
            # this creates the df for the time line for the report
            self.base_equi_df = equi_df
            self.base_equi_df.drop(self.stock_names, axis=1, inplace=True)
            self.base_equi_df[self.base_equi_df != 0] = np.nan
            self.base_equi_df[self.base_equi_df == 0] = 1

    def calc_equi(self, res):
        """
        calculates the equilbrium result for initialization
        first calculates the sum of the flows for each stock
        then calculates the sum of absolute sums
        this sum needs to be 0 for an equilibrium to exist

        :param res: OptimizeResult object from scipy.optimize.minimize
        :return: sum of absolute sums
        """
        tot_res = 0
        # iterates through level 1 of list of lists
        for expr in self.flow_expr_lst:
            st_res = 0
            # iterates through level 2 of the list of lists
            for i, el in enumerate(expr):
                # empty string needs to be in selection because if the first flow is negative, it will add an
                # empty string element to the expr (which is a list of strings)
                if el not in ['+', '-', '']:
                    out = res[el]
                    if expr[i - 1] == '-':
                        out = -out
                    # calculates the stock result
                    st_res += out
            tot_res += sum(abs(st_res))
        return tot_res

    def op_flows(self):
        """
        extracts the flows for the equilibrium calculation
        flow expressions are stored with the associated stocks in the stocks dataframe,
        thus having operations and names in the expression

        :return: list of lists with the flow expressions split up
        """
        self.flow_expr_lst = []
        for i, row in self.stocks.iterrows():
            flow_expr = row['flow expr']
            # split on + and - (only operations allowed in stocks) and keep the operator in the list
            flow_expr = re.split(r'([+-])', flow_expr)
            # strip all expressions to make sure that there are no errors due to spaces still in the thing
            flow_expr = [s.strip() for s in flow_expr]
            self.flow_expr_lst.append(flow_expr)

    def create_init_bounds(self):
        """
        # this has to go to equilbrium test
        incremental=True indicates that equilibria closer to the base run are searched,
        is more time intensive than incremental = false
        creates the initial bounds for the equilibrium function

        even with incremental = False equilibria can still found incrementally as even with very large max_equi bounds,
        there is the possibility that incrementally the bounds are increased, but it's unlikelier

        :return: list of tuples with the bounds for each exogenous variable in the model
        """

        self.bound_lst = []
        for i, name in enumerate(self.exo_names):
            if name in self.equi_excl:
                self.base_params.iloc[i] = self.equi_set[name][0]
        if self.set_inc:
            for i, value in self.base_params.iteritems():
                # if values are 0 at t0 they need to be manually set to an arbitrary bounds, otherwise they won't change
                # not sure how to set them effectively
                if self.exo_names[i] in self.equi_excl:
                    self.bound_lst.append((self.equi_set[self.exo_names[i]][0], self.equi_set[self.exo_names[i]][0]))
                else:
                    if value == 0:
                        self.bound_lst.append((0, 1))
                    else:
                        bounds = (value * (1 - self.sp), value * (1 + self.sp))
                        self.bound_lst.append(bounds)

    def build_bounds(self):
        """
        # this has to go to equilbrium test
        updates the bounds for each iteration of the solver
        method one increases the bounds based on the initial parameter value from the base run
        method two increases the bounds based on the result of the equilibrium function

        :return: updated bounds list, parameters for next iteration
        """
        if self.equi_method == 1:
            for i, var in enumerate(self.res_eq.x):

                if self.exo_names[i] not in self.equi_excl:
                    lb, ub = self.bound_lst[i]
                    # again we need to check if the initial is 0, then changed it to the result for bounds calculation
                    if self.base_params.loc[i] == 0:
                        # if initial parameter is zero, parameter is handled as if method 2
                        # even though method 1 is selected
                        # except that the applied space here is dependent on iter_cnt
                        value = var
                    else:
                        value = self.base_params.loc[i]
                    if lb == var:
                        lb = value * (1 - self.iter_cnt * self.sp)
                    elif ub == var:
                        ub = value * (1 + self.iter_cnt * self.sp)
                    if lb < self.equi_set[self.exo_names[i]][1]:
                        lb = self.equi_set[self.exo_names[i]][1]
                    if ub > self.equi_set[self.exo_names[i]][2]:
                        ub = self.equi_set[self.exo_names[i]][2]
                    self.bound_lst[i] = (lb, ub)
            self.equi_params = self.res_eq.x
        elif self.equi_method == 2:
            for i, var in enumerate(self.res_eq.x):
                if self.exo_names[i] not in self.equi_excl:
                    lb = var * (1 - self.sp)
                    ub = var * (1 + self.sp)
                    self.bound_lst[i] = (lb, ub)
            self.equi_params = self.res_eq.x
        else:
            pass

    def write_equi_to_doc(self, equi_dict):
        # this has to go to the equilibrium test
        """
        saves the equilbrium result to the doc file

        the equidict used here has all exogenous variables and for each either a number value, NE (No Equilbrium)
        , or BE (Bad Equlibrium)
        :param equi_dict: dictionary from the equilibrium test output, used to create the equi runs
        :return: saved .csv
        """
        for key, val in equi_dict.items():
            self.doc.loc[self.doc['Py Name'] == key, 'equi'] = val
        return self.save_csv('%s_doc' % self.out_name, self.doc, 'doc')

    def create_run_with_result(self, result):
        """
        creates a run with the results of some function, does not need to pass exo names because exo names are
        global in here
        :param result: list or series with parameter settings
        :return: df with the resulting run (endogenous variables)
        """
        run = Run('res_eq', 'equi', self.exo_names, result, self.endo_names)
        res = self.model.run(params=run.input_dict, return_columns=run.return_columns)
        run.run = res
        return run

    def check_equilibrium(self):
        """
        # this needs to go to equilibrium test
        this function checks the result of the equilibrium function and adjusts it if not all conditions for
        a good equilibrium are met
        if the sum for the equilibrium is 0, but the sum of all flows is 0, then an equilibrium was found,
        but it is just by setting all parameters to 0, thus making it impossible to use for other tests,
        thus the values are changed to BE, bad equilibrium

        if the result of the equilibrium function is larger than 0.1, then no equilibrium could be found, thus changing
        the values to NE, no equilibrium

        it is possible that no equilibrium is found because the while loop of the equilibrium function exists due to
        improvement being 0 even tough an equilibrium might be possible, but I don't know how to fix that
        :return: the updated dictionary with equi values (or NE, BE)
        """
        equi_dict = dict(zip(self.exo_names, self.res_eq.x))
        self.eq_res = 'GE'
        if self.eq_run.run[self.flow_names].iloc[0].sum(axis=0) == 0:
            for key, val in equi_dict.items():
                equi_dict[key] = 'BE'
                self.eq_res = 'BE'
        if self.res_eq.fun > 0.1:
            for key, val in equi_dict.items():
                equi_dict[key] = 'NE'
                self.eq_res = 'NE'
        return equi_dict

    def prepare_test(self):
        """
        For the equilibrium test there is no need for a run list as they are not passed through MP

        """
        if self.set_inc:
            self.create_init_bounds()
        self.res_lst = []
        self.equi_params = self.base_params
        self.iter_cnt = 1

    def run_test(self):
        """
        run test is the alternative for run with MP and collect res

        Equilibrium is currently the only test using it

        """
        # first optimizer run is executed to estalish a starting point
        # if not incremental, no bounds are necessary
        if self.set_inc:
            self.res_eq = opt.minimize(self.equilibrium, self.equi_params, bounds=self.bound_lst)
        else:
            self.res_eq = opt.minimize(self.equilibrium, self.equi_params)
        # results are gathered to document the initial search
        self.eq_run = self.create_run_with_result(self.res_eq.x)
        self.eq_run.calc_fit(self.base.run)
        self.res_lst.append((self.res_eq.fun, self.eq_run.fit, self.res_eq.x))
        # self improv is set to 1 to make sure it continues
        self.improv = 1
        while self.res_eq.fun > self.equi_res and self.improv > self.equi_iter:
            self.iter_cnt += 1
            # just a bit of reporting that things aren't hanging
            print('start', self.iter_cnt)
            if self.set_inc:
                # updating the bounds
                self.build_bounds()
                self.res_eq = opt.minimize(self.equilibrium, self.equi_params, bounds=self.bound_lst)
            else:
                self.res_eq = opt.minimize(self.equilibrium, self.equi_params)
            # gathering the results again
            self.eq_run = self.create_run_with_result(self.res_eq.x)
            self.eq_run.calc_fit(self.base.run)
            self.res_lst.append((self.res_eq.fun, self.eq_run.fit, self.res_eq.x))
            # calculates the difference between the last two iterations to set to equilibrium,
            # is -2 and -1 because the index in the list is one behind the count
            self.improv = self.res_lst[self.iter_cnt - 2][0] - self.res_lst[self.iter_cnt - 1][0]
            # if equilibrium is not found after 20 iterations, we should move on
            if self.iter_cnt == self.equi_maxiter:
                break
        self.model.create_full_df(self.base.run, self.test)
        self.model.add_run(self.eq_run.run, 'equilibrium run', self.test)
        # creating the full df to avoid issues with large dfs in MP (which is not the case here)
        pipe = Plass(self)
        for key, full_df in self.model.full_df_dict.items():
            pipe.save_csv(full_df, 'full_df', key)

    def save_output(self):
        """
        Saving the output from the equilibrium test

        """
        # this is all the output that doens't got through MP
        self.save_lst_csv(self.res_lst, 'equi_sum_%s' % self.out_name, 'equi',
                          columns=['equilibrium result', 'error to base', 'parameters'], append=False)
        # this is for the search of equilibrium conditions in the base and equi run
        self.sum_df = pd.DataFrame(columns=['name', 'start', 'end'].extend(self.stock_names))
        self.src_equi(self.base.run, 'base')
        self.src_equi(self.eq_run.run, 'equi')
        # sum df could be empty if no equilibrium condition has been found
        if not self.sum_df.empty:
            order = ['name', 'start', 'end']
            order.extend(self.stock_names)
            self.sum_df = self.sum_df[order]
        self.sum_df.to_csv(os.path.join(self.folder_dict[self.test], 'equi_sum.csv'))

        exo_r_dict = self.check_equilibrium()
        # testing feature to compare the found equilibria between models
        equi_rep = [[self.res_eq.fun, self.eq_run.fit, self.res_eq.x, self.iter_cnt]]
        equi_db = pd.DataFrame(equi_rep)
        with open(os.path.join(self.folder, 'equidoc.csv'), 'a') as f:
            equi_db.to_csv(f, header=False)

        self.write_equi_to_doc(exo_r_dict)
        pipe = Plass(self)
        # since equi is not going through MP, the model creation is called here a bit differently
        pipe.create_model('equi', self.model.full_df_dict['equi'], self.test)
        pipe.create_timeline(self.base_equi_df, 'equi_base')

    def write_report(self):
        """
        writing the report for the equilibrium test
        """
        rep = Report(self.folder, self.file)
        # we don't need the its here, but we need to get rid of the pickle file
        f_path = os.path.join(self.folder_dict[self.test], 'endo_its.pickle')
        os.remove(f_path)
        equi_doc = self.doc.loc[self.doc['equi'].notnull()]
        # report tuple includes section title, equilibrium result, equilibrium settings,
        # list with equilibrium conditions, link to test source
        rep_tpl = (self.class_name, self.eq_res, equi_doc, self.sum_df,
                   self.folder_dict[self.test].replace(self.folder_dict['source'], '').lstrip('\\'))
        rep.write_equi(rep_tpl)
        rep.save_report()


class TimeStep(Test):
    """
    Timestep test for the testing battery
    """
    def __init__(self, folder, file_name):
        super(TimeStep, self).__init__(folder, file_name, 'timestep')
        self.err_list = []
        self.MP = True
        self.cf = ConfigParser()
        # config folder doesn't change, so it's put here to it's on one line
        self.cf.read(os.path.join(os.path.split(folder)[0], '_config', 'settings.ini'))
        self.start_ts = self.cf['tests'].getfloat('ts_start', fallback=1)
        self.step_ts = self.cf['tests'].getfloat('ts_iter', fallback=0.5)
        self.step_cnt = self.cf['tests'].getint('ts_maxiter', fallback=10)
        self.ts_threshold = self.cf['tests'].getfloat('ts_threshold', fallback=0.015)
        self.class_name = 'TimeStep'

    def prepare_test(self):
        """
        prepares the runs for this test
        """
        rts = np.arange(self.base_builtin.iloc[1], self.base_builtin.iloc[0] + 1, 1)
        base_full = self.model.run(return_timestamps=rts, reload=True)
        col_lst = list(base_full)
        for col in col_lst:
            if base_full[col].all() == 0:
                base_full[col] = np.nan
                # endos that are always zero could be added to the report at some point 17.07.18/sk
        self.base.add_run(base_full[self.endo_names])
        self.model.create_full_df(self.base.run, 'timestep')
        for i in range(self.step_cnt):
            ts = self.start_ts * self.step_ts ** i
            name = 'timestep_%s' % ts
            # Run has inputs name,full_ID,exo_names=None,params=None,return_columns=None
            self.run_lst.append(Run(name, 'timestep', [self.builtin_names.iloc[-1]], [ts], self.endo_names,
                                    'TimeStep=%s' % ts, rts, reload=True))

    def save_output(self):
        """
        saving the output for the time step test
        """

        # this is all the output that doens't got through MP
        res_lst = []
        # tracklist is just for testing purposes
        trck_lst = []
        comp_df = self.model.full_df_dict['timestep']
        comp_df = comp_df.loc(axis=1)[:, self.stock_names]
        base_name = 'base_%s' % self.base_builtin.iloc[-1]
        res_lst.append((base_name, 1))
        for i in range(1, self.step_cnt):
            ts = self.start_ts * self.step_ts ** i
            sm_name = 'timestep_%s' % ts
            lg_name = 'timestep_%s' % (ts * 2)
            sens_df = comp_df.loc(axis=1)[[sm_name, lg_name], :]
            sens_df = sens_df.copy()
            # dropna should be deleted 17.07.18/sk
            # sens_df.dropna(inplace=True)
            if (sens_df.isnull().sum(axis=1) == 0).all():
                # absolute value is taken because we only care about the distance to the upper run
                sens_df = abs(
                    (sens_df.loc(axis=1)[sm_name] - sens_df.loc(axis=1)[lg_name]) / sens_df.loc(axis=1)[lg_name])
                est = sens_df.mean(axis=0).mean(axis=0)
            else:
                est = 1
            res_lst.append((lg_name, est))
        for i, step in enumerate(res_lst[1:]):
            name, est = step
            if est <= self.ts_threshold:
                ts = name.split('_')[-1]
                trck_lst.append((self.out_name, self.base_builtin.iloc[-1], ts, est))
                self.ts_rep = (self.out_name, self.base_builtin.iloc[-1], ts, est)
                self.save_lst_csv(trck_lst, 'ts_tracking', 'source',
                                  ['Model Name', 'Actual TS', 'Optimal TS', 'Opt Result'], append=True)
                break
            # the last element is i=8 because we don't use the first time step for iteration
            elif i == 8:
                # if it doesn't find the optimal timestep, we report a 'NF' for not found
                trck_lst.append((self.out_name, self.base_builtin.iloc[-1], 'NF', est))
                self.ts_rep = (self.out_name, self.base_builtin.iloc[-1], 'NF', est)
                self.save_lst_csv(trck_lst, 'ts_tracking', 'source',
                                  ['Model Name', 'Actual TS', 'Optimal TS', 'Opt Result'], append=True)
                break
        self.save_lst_csv(res_lst, 'result', self.test, ['Timestep', 'Result'], append=False)

    def write_report(self):
        """
        write the report for the time step test
        """
        rep = Report(self.folder, self.file)
        # we have to pickle this because with MP, the passing of arguments is faulty
        # the endo_its is not needed here, but still needs to be removed
        f_path = os.path.join(self.folder_dict[self.test], 'endo_its.pickle')
        os.remove(f_path)
        rep_tpl = (
            self.class_name, self.ts_rep,
            self.folder_dict[self.test].replace(self.folder_dict['source'], '').lstrip('\\'))
        rep.write_tstep(rep_tpl)
        rep.save_report()


class Switches(Test):
    """
    testing the different switch settings in all the combinations
    """
    def __init__(self, folder, file_name):
        super(Switches, self).__init__(folder, file_name, 'switches')
        self.err_list = []
        self.MP = True
        self.class_name = 'Switches'
        self.condensed = False

    def create_switch_settings(self):
        """
        # this needs to go to the switches test
        creates the df with switch settings

        condensed only returns the switch settings where all are turned on or turned off

        :return:
        """
        self.switch_lst = []
        for i, row in self.switches.iterrows():
            self.switch_lst.append(row['Py Name'])
        self.nmb_switch = len(self.switch_lst)
        if self.nmb_switch > 0:
            set_switch = [np.reshape(np.array(i), (1, self.nmb_switch)) for i in
                          itertools.product([0, 1], repeat=self.nmb_switch)]
            self.switch_df = pd.DataFrame(data=np.reshape(set_switch, (2 ** self.nmb_switch, self.nmb_switch)),
                                          columns=self.switch_lst)
            if self.condensed:
                self.switch_df = self.switch_df.loc[self.switch_df.sum(axis=1).isin([0, self.nmb_switch])]
        else:
            self.switch_df = pd.DataFrame()
        self.save_csv('switch_settings', self.switch_df, self.test)

    def prepare_test(self):
        """
        prepare the switcehs test
        """
        self.create_switch_settings()
        self.model.create_full_df(self.base.run, 'full')
        self.model.create_full_df(self.base.run, 'sum')
        for i, row in self.switch_df.iterrows():
            name = 'switch_run_%s' % i
            self.run_lst.append(Run(name, 'full', row.index, row.values, self.endo_names))
            if row.sum() == 1:
                self.run_lst.append(Run(name, 'sum', row.index, row.values, self.endo_names))

    # maybe the endo plots don't need to be quite so numerous here... maybe just the stocks
    def write_report(self):
        """
        write the report for the switches test
        """
        rep = Report(self.folder, self.file)
        # we have to pickle this because with MP, the passing of arguments is faulty
        f_path = os.path.join(self.folder_dict[self.test], 'endo_its.pickle')
        pickle_in = open(f_path, 'rb')
        endo_its = pickle.load(pickle_in)
        pickle_in.close()
        os.remove(f_path)
        rep_tpl = (self.class_name, self.switch_df, endo_its,
                   self.folder_dict[self.test].replace(self.folder_dict['source'], '').lstrip('\\'))
        rep.write_swit(rep_tpl)
        rep.save_report()


class Distance(Test):
    """
    the distance test of the tb

    currently somewhat faulty and only available in testing mode

    also has no setting in the config file
    """
    def __init__(self, folder, file_name):
        super(Distance, self).__init__(folder, file_name, 'distance')
        self.err_list = []
        self.MP = False
        self.class_name = 'Distance'
        # needs to be verified
        # need all functions that contain a stock
        self.stocklike_functions = ['DELAY1', 'DELAY1I', 'DELAY3', 'DELAY3I', 'DELAY N',
                                    'SMOOTH', 'SMOOTHI', 'SMOOTH3', 'SMOOTH3I', 'SMOOTH N']
        self.cf = ConfigParser()
        # config folder doesn't change, so it's put here to it's on one line
        self.cf.read(os.path.join(os.path.split(folder)[0], '_config', 'settings.ini'))
        self.dist_maxiter = self.cf['tests'].getint('dist_maxiter', fallback=20)

    def create_emtpy_matrix(self):
        """
        create an NxN matrix full with np.nan

        :return: df, empty matrix
        """
        dm = np.empty((len(self.var_lst), len(self.var_lst)))
        dm[:] = np.nan
        self.dist_matrix = pd.DataFrame(dm)
        self.dist_matrix.columns = self.var_lst
        self.dist_matrix['name'] = self.var_lst
        self.dist_matrix.set_index('name', inplace=True)

    def make_loopdoc(self):
        """

        :return:
        """
        loop_doc = self.doc.copy()
        for i, row in loop_doc.iterrows():
            row = row.copy()
            els = row['elements']
            els = [x for x in els if not self.constant(x)]
            if 'table_expr' in els:
                els = []
            loop_doc.at[i, 'elements'] = els
        return loop_doc

    def loop_tree(self, in_lst):
        """

        :param in_lst:
        :return:
        """
        loop_doc = self.make_loopdoc()
        new_level = []
        i = 0
        for lst in in_lst[i]:
            # then we add the elements from the stocks as the first level
            for var in lst:
                n_lst = loop_doc.loc[loop_doc['Real Name'] == var]['elements'].iloc[0]
                r_lst = loop_doc.loc[loop_doc['Real Name'] == var]['init elements'].iloc[0]
                f_lst = [x for x in n_lst if x not in r_lst]
                new_level.append(f_lst)
            in_lst.append(new_level)
        while True:
            # then we iterate through the lists making a new list for each level of the
            # length of the sum of elements of the previous level
            i += 1
            new_level = []
            for lst in in_lst[i]:
                if type(lst) == list:
                    for var in lst:
                        if var not in self.stock_names:
                            if not self.constant(var):
                                n_lst = loop_doc.loc[loop_doc['Real Name'] == var]['elements'].iloc[0]
                                if n_lst:
                                    new_level.append(n_lst)
                                else:
                                    new_level.append(np.nan)
                            else:
                                new_level.append(np.nan)
                        else:
                            new_level.append(np.nan)
                else:
                    # for every loop that is already finished, there needs to be a nan added to keep the length correct
                    new_level.append(np.nan)

            try:
                # when all loops have finished, we break the while loop
                if np.isnan(new_level).all():
                    return in_lst, i
            except:
                pass
            # this is just avoid infinite loops, not sure what the threshold should be 19.06.18/sk
            if i == self.dist_maxiter:
                return in_lst, i

            # each new level is added, the last level with all nan is not added
            in_lst.append(new_level)
            loop_df = pd.DataFrame(in_lst)
            loop_df.to_csv(os.path.join(self.folder_dict[self.test], 'level%s.csv' % i))

    def loop_explore(self, in_lst, src_lst, level, max_level):
        """

        :param in_lst:
        :param src_lst:
        :param level:
        :param max_level:
        :return:
        """
        out_lst = []
        if level <= max_level:
            for j, lst in enumerate(src_lst[level]):
                if type(lst) == list:
                    for var in lst:
                        t_lst = in_lst[j].copy()
                        t_lst.append(var)
                        out_lst.append(t_lst)
                else:
                    t_lst = in_lst[j].copy()
                    t_lst.append(np.nan)
                    out_lst.append(t_lst)
            level += 1
            return self.loop_explore(out_lst, src_lst, level, max_level)
        else:
            return in_lst

    @staticmethod
    def make_loopdict(in_lst):
        """

        :param in_lst:
        :return:
        """
        loop_dict = {}
        for lst in in_lst:
            if lst[0] != lst[-1]:
                key = lst[0]
                if key in loop_dict:
                    loop_dict[key].append(lst)
                else:
                    loop_dict[key] = [lst]
        return loop_dict

    def loop_combine(self, in_lst, loop_lst, loop_dict, iteration=0):
        """

        :param in_lst:
        :param loop_lst:
        :param loop_dict:
        :param iteration:
        :return:
        """
        out_lst = []
        t_lst = []
        for lst in in_lst:
            # first we move the loops that are loops already to the loop list
            if lst[0] == lst[-1]:
                loop_lst.append(lst)
            # then we move the loop elements that are not yet loops to a temporary list
            # also we build the dict with the different starting points (stocklike vars)
            else:
                t_lst.append(lst)
        if t_lst:
            stock_lst = list(loop_dict.keys())
            visited_lst = [stock_lst[0]]
            for stock in stock_lst[1:]:
                for lst in t_lst:
                    if lst[-1] not in visited_lst:
                        # this is to avoid infinite loops where the first loop element can only be completed
                        # by a loop of two other stocks
                        if lst.count(lst[-1]) < 2:
                            for el in loop_dict[lst[-1]]:
                                b_lst = lst.copy()
                                b_lst.extend(el[1:])
                                out_lst.append(b_lst)
                visited_lst.append(stock)
            iteration += 1
            print(iteration)
            return self.loop_combine(out_lst, loop_lst, loop_dict, iteration)
        else:
            return loop_lst

    @staticmethod
    def clean_looplst(in_lst, stock_lst):
        """

        :param in_lst:
        :param stock_lst:
        :return:
        """
        out_lst = []
        for lst in in_lst:
            # cleaning out the np.nan from the list to arrive at the loop building blocks
            lst = [x for x in lst if not pd.isnull(x)]
            out_lst.append(lst)
        # then we remove the loop elements that don't end in a stocklike variable, because they are dead ends
        out_lst = [x for x in out_lst if x[-1] in stock_lst]
        return out_lst

    @staticmethod
    def clean_loops(in_lst):
        """

        :param in_lst:
        :return:
        """
        out_lst = []
        for lst in in_lst:
            t_lst = lst[1:]
            out_lst.append(t_lst)
        # out_lst = [x[::-1] for x in out_lst]
        return out_lst

    def run_test(self):
        """
        run the distance test
        """
        self.var_lst = []
        const_names = self.const['Real Name'].tolist()
        self.var_lst.extend(const_names)
        self.var_lst.extend(self.endo_names)
        self.exp_lst = [x for x in self.endo_names if x not in self.stock_names]
        self.create_emtpy_matrix()

        for var in self.var_lst:
            interval = 0
            alevel_lst = []
            olevel_lst = []
            alevel_lst.append(var)
            self.dist_matrix.loc[var, var] = interval
            while len(alevel_lst) != 0:
                # adding the next level variables in a new list to make sure we iterate properly
                olevel_lst.extend(alevel_lst)
                nlevel_lst = []
                for el in alevel_lst:
                    if el in self.exp_lst:
                        if el in self.flow_names:
                            nlevel_lst.extend(self.flows.loc[self.flows['Real Name'] == el]['elements'].iloc[0])
                        else:
                            nlevel_lst.extend(self.endo.loc[self.endo['Real Name'] == el]['elements'].iloc[0])
                    elif el in self.stock_names:
                        nlevel_lst.extend(self.stocks.loc[self.stocks['Real Name'] == el]['elements'].iloc[0])
                # removing variables of types that we don't care about, e.g. tables
                nlevel_lst = [x for x in nlevel_lst if x in self.var_lst]
                # removing variables we have visited before, to avoid loops
                # this means that the distance in the matrix is the shortest available between two variables
                nlevel_lst = [x for x in nlevel_lst if x not in olevel_lst]
                alevel_lst = nlevel_lst
                interval += 1
                # writing the distance into the matrix
                for el in alevel_lst:
                    self.dist_matrix.loc[el, var] = interval

        self.dist_matrix = self.dist_matrix[self.dist_matrix.columns[self.dist_matrix.sum() != 0]]
        output_vars = self.dist_matrix.loc[self.dist_matrix.sum(axis=1) == 0].index.tolist()
        lst = list(self.dist_matrix.columns)
        lst = lst[::-1]
        self.dist_matrix.sort_values(by=lst, inplace=True)

        self.save_lst_csv(output_vars, 'output_vars_%s' % self.out_name, self.test, append=False)
        self.save_csv('dist_matrix_%s' % self.out_name, self.dist_matrix, self.test)

        loop_lst = []
        stocklike_lst = []
        # we start the loop list with the stocks because every loop has to have a stock
        # we still need to add the stocklike items to the starting list 18.06.18/sk
        stocklike_lst.extend(self.stock_names)
        for i, row in self.doc.iterrows():
            if [x for x in row['function list'] if x in self.stocklike_functions]:
                stocklike_lst.append(row['Real Name'])
        loop_lst.append([stocklike_lst])
        print('start')

        loop_lst, max_iteration = self.loop_tree(loop_lst)
        print('tree done')

        # loop database needs to be initiated as an empty list of lists
        loop_db = [[]]
        # loop explore takes the elements tree and makes the loop sequences
        # right now we're just looking at stocklike to stocklike connections
        loop_out = self.loop_explore(loop_db, loop_lst, 0, max_iteration)
        print('explore done')
        loop_out = self.clean_looplst(loop_out, stocklike_lst)

        loop_final = []
        loop_dict = self.make_loopdict(loop_out)
        loop_final = self.loop_combine(loop_out, loop_final, loop_dict)
        loop_final = self.clean_loops(loop_final)
        loop_df = pd.DataFrame(loop_final)
        cols = list(loop_df)
        loop_df = loop_df.sort_values(by=cols, na_position='first')
        loop_df.drop_duplicates(inplace=True)
        loop_df.to_csv(os.path.join(self.folder_dict[self.test], 'loopfinal.csv'))

    def run_test_mp(self, run=None):
        """
        this should not be necessary 190818/sk
        :param run:
        """
        pass

    def collect_res(self, res):
        """
        this should not be necessary 190818/sk
        :param res:
        """
        pass

    def save_full_output_mp(self, args):
        """

        :param args:
        """
        pass


class KnockOut(Test):
    """
    knockout test of the testing battery

    currently faulty and only available in the testing mode

    has no setting in the config file
    """
    def __init__(self, folder, file_name):
        super(KnockOut, self).__init__(folder, file_name, 'knockout')
        self.err_list = []
        self.MP = True
        self.class_name = 'KnockOut'

    def create_ko_lst(self):
        """
        creating the knockout list

        this is not correct and needs to be reworked 30.07.18/sk
        """
        self.rn_lst = []
        self.ko_lst = []
        for i, row in self.flows.iterrows():
            self.rn_lst.extend(row['elements'])
            # the flows themselves are also added to make sure we cover flows where it's stock/delay, where it raises
            # an error and won't process it
            # downside is that we might have some double knockouts, but that's not too much of a problem
            self.rn_lst.append(row['Real Name'])
        # here we do the rn_lst with real names, they are afterwards converted
        self.rn_lst = [x for x in self.rn_lst if x not in self.stock_names]
        self.rn_lst = [x for x in self.rn_lst if not self.constant(x)]
        self.rn_lst = [x for x in self.rn_lst if x not in ['Time', 'TIME STEP']]
        # the switches are removed from the list because they are handled in the switches test
        self.rn_lst = [x for x in self.rn_lst if x not in self.switches.loc(axis=1)['Real Name'].tolist()]
        for var in self.rn_lst:
            self.ko_lst.append(self.doc[self.doc['Real Name'] == var]['Py Name'].iloc[0])

    def prepare_test(self):
        """
        prepare the knockout test
        """
        self.create_ko_lst()

        for var in self.ko_lst:
            name = '%s_%s' % (var, 0)
            full_id = self.doc.loc[self.doc['Py Name'] == var]['Real Name'].iloc[0]
            self.model.create_full_df(self.base.run, full_id)
            # Run has inputs name,full_ID,exo_names=None,params=None,return_columns=None
            self.run_lst.append(Run(name, full_id, [var], [0], self.endo_names,
                                    '%s=%s' % (var, 0), reload=True))

    def write_report(self):
        """
        write the report for the knockout test
        """
        rep = Report(self.folder, self.file)
        # we have to pickle this because with MP, the passing of arguments is faulty
        # the endo_its is not needed here, but still needs to be removed
        f_path = os.path.join(self.folder_dict[self.test], 'endo_its.pickle')
        os.remove(f_path)
        rep_tpl = (
            self.class_name, self.rn_lst,
            self.folder_dict[self.test].replace(self.folder_dict['source'], '').lstrip('\\'))
        rep.write_ko(rep_tpl)
        rep.save_report()


class Extreme(Test):
    """
    extreme condition test of the test battery

    currently the best developed test

    however the choice of extreme values should be improved
    """
    def __init__(self, folder, file_name, max_extreme):
        super(Extreme, self).__init__(folder, file_name, 'extreme')
        self.err_list = []
        self.MP = True
        self.max = max_extreme
        self.tbl_err_lst = []
        self.class_name = 'Extreme'
        self.tbl_lst = []

    @staticmethod
    def read_tbl(table_expr):
        """

        :param table_expr:
        :return:
        """
        table_expr = table_expr.replace(' ', '')
        exp = re.sub('\[.*\]', '', table_expr)
        exp = re.sub('\(,', '', exp)
        # in table functions of WITH LOOKUP, there are three closing brackets still in the table_expr
        exp = re.sub('\)\)\)', ')', exp)
        # in regular table functions there are only two closing brackets,
        # so this takes effect if the previous line doesn't
        exp = re.sub('\)\)', ')', exp)
        pair_lst = re.split(',\s*(?![^()]*\))', exp)
        # returns list of strings
        return pair_lst

    @staticmethod
    def chk_monotonic(lst):
        """

        :param lst:
        :return:
        """
        dlst = np.diff(lst)
        return np.all(dlst <= 0) or np.all(dlst >= 0)

    def op_tbl(self, lst, name, i):
        """

        :param lst:
        :param name:
        :param i:
        """
        ylist = []
        xlist = []
        orig_chk = False
        # y_orig is the list of origin points, there could be multiple
        y_orig = []
        for pair in lst:
            x, y = literal_eval(pair)
            if x == 1 and y == 1:
                orig_chk = True
                y_orig.append(y)
            elif x == 1 and y == 0:
                orig_chk = True
                y_orig.append(y)
            elif x == 0 and y == 1:
                orig_chk = True
                y_orig.append(y)
            elif x == 0 and y == 0:
                orig_chk = True
                y_orig.append(y)
            xlist.append(x)
            ylist.append(y)
        # there should also be a test to see if the output ranges between 0 and 1,
        # which then would be acceptable for max values 02.07.18/sk
        if not orig_chk:
            # table errors should not be reported in the model errors,
            # they are in the report as table errors 21.06.18/sk
            # self.model.err_lst.append((self.test_name, 'Missing Point (1,1)',
            # 'Table Formulation Error', name, '', ''))
            self.tbl_err_lst.append(
                (self.test_name, 'Missing Point (0,0), (1,1), (0,1) or (1,0)', 'Table Formulation Error', name, '', ''))
        if not self.chk_monotonic(ylist):
            # self.model.err_lst.append((self.test_name, 'Table not monotonic',
            # 'Table Formulation Error', name, '', ''))
            self.tbl_err_lst.append((self.test_name, 'Table not monotonic', 'Table Formulation Error', name, '', ''))
        var_lst = self.get_rec_function(name)
        # creating the test list, to keep in order it's first min, max then all points of origin that are not min or max
        y_test_lst = [min(ylist), max(ylist)]
        for y in y_orig:
            if y not in y_test_lst:
                y_test_lst.append(y)
        self.tbl_lst.append([i, name, var_lst, y_test_lst])

    def get_rec_function(self, name):
        """

        :param name:
        :return:
        """
        rec_lst = []
        # endo doesn't include flows
        for i, row in self.endo.iterrows():
            if name in row['elements']:
                rec_lst.append(row['Py Name'])
        for i, row in self.flows.iterrows():
            if name in row['elements']:
                rec_lst.append(row['Py Name'])
        return rec_lst

    def prepare_test(self):
        """
        prepare the extreme condition test runs
        """
        if len(self.tables.index) > 0:
            for i, row in self.tables.iterrows():
                # DATATABLE is the indicator for data input in table form, if not named, the table will be checked
                # PySD helper prepares data in this way
                if 'DATATABLE' not in row['Real Name']:
                    if row['table expr'].startswith('([('):
                        self.model.create_full_df(self.base.run, 'table%s' % i)
                        lst = self.read_tbl(row['table expr'])
                        self.op_tbl(lst, row['Real Name'], i)
            for tbl in self.tbl_lst:
                for var in tbl[2]:
                    for param in tbl[3]:
                        name = '%s_%s' % (var, param)
                        # currently there is one full df per table, which is inconsistent with for example sensitivity
                        # need to evaluate which approach is better (one model with all runs or
                        # multiple models with few runs)
                        self.run_lst.append(Run(name, 'table%s' % tbl[0], [var], [param], self.endo_names,
                                                '%s_Output=%s' % (tbl[1], param), reload=True))

        ep_lst = [0, self.max]
        # value dict reports the extreme values to ensure that they are extreme
        ext_value_dict = {}
        for ep in ep_lst:
            self.model.create_full_df(self.base.run, 'mult%s' % ep)
            for i, row in self.const.iterrows():
                name = '%s_%s' % (row['Real Name'], ep)
                w_params = self.base_params.copy()
                w_params.iloc[i] *= ep
                if ep != 0:
                    ext_value_dict[row['Real Name']] = [0, w_params.iloc[i]]
                # Run has inputs name,full_ID,exo_names=None,params=None,return_columns=None
                self.run_lst.append(Run(name, 'mult%s' % ep, self.exo_names, w_params, self.endo_names,
                                        '%s=%s' % (row['Real Name'], w_params.iloc[i]), reload=True))
        self.ext_value_df = pd.DataFrame(ext_value_dict)
        self.ext_value_df = self.ext_value_df.transpose()
        # this try block is just to avoid errors if there are no extreme runs (which should not happen)
        try:
            self.ext_value_df.columns = ['LB', 'UB']
        except:
            pass

    def save_output(self):
        """
        save the output for the extreme condition test
        """
        # flagging tables if a run could not be executed
        tbl_flag = []
        if self.tbl_lst:
            for tbl in self.tbl_lst:
                tbl_df = self.model.full_df_dict['table%s' % tbl[0]]
                # here we drop columns that have a np.nan, because we want to be sure that
                # table functions work for the entire time horizon
                executed = list(tbl_df.dropna(axis=1).columns.levels[0])[1:]
                tbl.append(executed)
                if len(tbl[2]) * len(tbl[3]) != len(executed):
                    tbl_flag.append(tbl)
            self.save_lst_csv(self.tbl_lst, 'table_summary', self.test,
                              columns=['Table ID', 'Table Name', 'Uses', 'Output Tested', 'Runs Executed'])
        self.tbl_flag_df = pd.DataFrame(tbl_flag)
        if not self.tbl_flag_df.empty:
            self.tbl_flag_df.columns = ['Table ID', 'Table Name', 'Uses', 'Output Tested', 'Runs Executed']
        flagged = []
        for run in self.run_lst:
            for var in run.var_dict['only neg']:
                if var in self.base.var_dict['only pos']:
                    flagged.append((run.name, var, 'unexpected negative values'))
            for var in run.var_dict['pos and neg']:
                if var in self.base.var_dict['only pos']:
                    flagged.append((run.name, var, 'unexpected negative values'))
        self.flag_df = pd.DataFrame(flagged)
        if not self.flag_df.empty:
            self.flag_df.columns = ['Run', 'Variable', 'Flag Description']
        if self.testing_mode:
            self.flag_df.to_csv(os.path.join(self.folder_dict[self.test], 'flagged.csv'))

    def write_report(self):
        """
        write the report for the extreme condition test
        """
        rep = Report(self.folder, self.file)
        # we have to pickle this because with MP, the passing of arguments is faulty
        f_path = os.path.join(self.folder_dict[self.test], 'endo_its.pickle')
        pickle_in = open(f_path, 'rb')
        endo_its = pickle.load(pickle_in)
        pickle_in.close()
        os.remove(f_path)
        rep_tpl = (self.class_name, self.max, endo_its, self.tbl_lst, self.tbl_err_lst, self.flag_df, self.ext_value_df,
                   self.tbl_flag_df, self.folder_dict[self.test].replace(self.folder_dict['source'], '').lstrip('\\'))
        rep.write_ext(rep_tpl)
        rep.save_report()


class Horizon(Test):
    """
    tests different time horizon settings for the model
    """
    def __init__(self, folder, file_name):
        super(Horizon, self).__init__(folder, file_name, 'horizon')
        self.err_list = []
        self.MP = True
        self.hor_lst = [(0, 3), (1, 3), (1, 2), (2, 3)]
        self.class_name = 'Horizon'

    def prepare_test(self):
        """
        preparing the run list for the horizon test
        """
        self.model.create_full_df(self.base.run, 'horizon')
        for lims in self.hor_lst:
            init, final = lims
            wparams = self.base_builtin.copy()
            wparams.iloc[1] = self.base_builtin.iloc[1] + init * (self.base_builtin.iloc[0] - self.base_builtin.iloc[1])
            wparams.iloc[0] = self.base_builtin.iloc[1] + final * (
                    self.base_builtin.iloc[0] - self.base_builtin.iloc[1])
            name = 'horizon%s_%s' % (wparams.iloc[1], wparams.iloc[0])
            # Run has inputs name,full_ID,exo_names=None,params=None,return_columns=None
            self.run_lst.append(Run(name, 'horizon', self.builtin_names, wparams, self.endo_names))

    def write_report(self):
        """
        write the report for the horizon test
        """
        rep = Report(self.folder, self.file)
        # we have to pickle this because with MP, the passing of arguments is faulty
        # the endo_its is not needed here, but still needs to be removed
        f_path = os.path.join(self.folder_dict[self.test], 'endo_its.pickle')
        os.remove(f_path)
        rep_tpl = (self.class_name, self.folder_dict[self.test].replace(self.folder_dict['source'], '').lstrip('\\'))
        rep.write_hori(rep_tpl)
        rep.save_report()
