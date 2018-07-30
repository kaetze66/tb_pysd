"""
base test includes all methods that are required for all the tests
also includes dummy methods for initialize, prepare, run, save_output for tests that don't need that

needs to be single thread and MP compatible

tests have 4 stages
- initialize
- prepare
- run
- save output

in MP runs, run is separated in run and collect (due to MP function)

Version 0.2
Update 30.07.18/sk
"""

import os
import pandas as pd
from timeit import default_timer as timer
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tb.tb_backend.model import AugModel
from tb.tb_backend.savingpipe import Plass
from tb.tb_backend.run import Run
from ast import literal_eval
from configparser import ConfigParser


class Test:
    """
    defines all methods that are common for tests

    all tests then just have to write the methods specific to them

    Has the following parts:

    - Folder operations
    - Tools for saving and operations
    - Doc file operations
    - General test operations

    todo:
    - pipes are opened all over the place, maybe one single saving pipe is better for performance

    """

    def __init__(self, folder, file_name, test_name):
        self.folder = folder
        self.file = os.path.join(folder, file_name)
        self.file_name = file_name
        self.out_name = file_name.split('.')[0]
        # folder dict could be done here, but currently it's filled later
        self.folder_dict = {}
        self.test_name = test_name
        self.model = AugModel(self.file)
        # doc will be filled later
        self.doc = None
        # run list will be generated later
        self.run_lst = []
        self.run_coll = []
        # variable lists for building the models
        self.node_lst = []
        self.edge_lst = []
        # new tests need to be added here
        self.folder_type_dict = {'sensitivity': 'sens', 'equilibrium': 'equi', 'translation': 'doc', 'distance': 'dist',
                                 'switches': 'swit', 'timestep': 'tstep', 'montecarlo': 'mc', 'knockout': 'ko',
                                 'extreme': 'ext', 'horizon': 'hori'}
        self.test = self.folder_type_dict[test_name]
        self.err_file_name = '%s_error_file.txt' % test_name
        # config file is defined and read in here
        # this doesn't need to be an attribute 30.07.18/sk
        self.cf = ConfigParser()
        # config folder doesn't change, so it's put here to it's on one line
        self.cf.read(os.path.join(os.path.split(folder)[0], '_config', 'tb_config.ini'))
        self.testing_mode = self.cf['testing'].getboolean('testing_mode', fallback=False)
        self.full_df_output = self.cf['testing'].getboolean('full_df_output', fallback=True)
        self.cf.read(os.path.join(os.path.split(folder)[0], '_config', 'settings.ini'))
        self.node_width = self.cf['basetest'].get('node_width', fallback='2')
        self.node_height = self.cf['basetest'].get('node_height', fallback='1.2')
        # test with sens is just to avoid having to calculate the sens and norm runs for every test
        self.test_with_sens = ['sens']

    # Folder Operations #
    def make_folder_dict(self):
        """
        General folder structure (non model specific)

        """
        spl_folder = os.path.split(self.folder)
        # report folder could be moved to the source folder, would make copying files a bit more simple
        self.folder_dict = {'source': self.folder,
                            'report': os.path.join(spl_folder[0], '_report', spl_folder[1]),
                            'flag': os.path.join(spl_folder[0], '_report', spl_folder[1], '_flag'),
                            'trans_debug': os.path.join(spl_folder[0], '_report', spl_folder[1], '_debug',
                                                        '_translation'),
                            'run_debug': os.path.join(spl_folder[0], '_report', spl_folder[1], '_debug', '_runtime'),
                            'run_pipe': os.path.join(spl_folder[0], 'run_pipe'),
                            'graphviz': os.path.join(spl_folder[0], 'utilities', 'graphviz')}
        self.folder_dict['base'] = os.path.join(self.folder_dict['source'], self.out_name)
        # new tests need to be added here
        for key, val in self.folder_type_dict.items():
            output_folder = os.path.join(self.folder_dict['source'], self.out_name, val)
            self.folder_dict[val] = output_folder

    def initiate_folder(self):
        """
        makes the output folder if it doesn't exist
        cleans it if not empty

        is within __main__ if MP is on to avoid any overwriting of files from initialization

        """
        try:
            os.makedirs(self.folder_dict[self.test])
        except OSError:
            pass
        # deletes all files in the active folder to avoid confusion
        olfiles = [f for f in os.listdir(self.folder_dict[self.test])]
        for file in olfiles:
            os.remove(os.path.join(self.folder_dict[self.test], file))

    # tools for saving and operations #
    @staticmethod
    def constant(expr):
        """
        testing if an expression is numeric

        :param expr: any expression to be tested
        :return: true if numeric, false if not numeric
        """
        try:
            float(expr)
            return True
        except ValueError:
            return False

    def move_mdl_debug(self, debug_type):
        """
        this moves the models to debug for further evaluation
        source is always source folder, as models are always in the source folder

        :param debug_type: string, can be 'flag', 'trans_debug' or 'run_debug'
        :return: none
        """
        shutil.copy2(self.file, self.folder_dict[debug_type])

    # Doc file operations #
    def read_doc_file(self):
        """
        reads in variables from the doc file created in translation
        doc file is more detailed than the regular vensim2py as it distinguishes the types

        if items in doc file could be read from pysd, then the testing battery would make a big step towards
        being compatible with xmile

        :return: dataframe from the doc file
        """
        doc_file = '%s_doc.csv' % self.out_name
        self.doc = pd.read_csv(os.path.join(self.folder_dict['doc'], doc_file), index_col=0)
        self.identify_switches()
        liteval_lst = ['elements', 'function list', 'init elements']
        for i, row in self.doc.iterrows():
            row = row.copy()
            for col in liteval_lst:
                lst = literal_eval(row[col])
                self.doc.at[i, col] = lst

    def identify_switches(self):
        """
        this is necessary since for descriptives the switches are counted as constants
        Switches will be treated only as binary switches with 0 and 1 as permissible settings
        Switch settings will be treated as different models, i.e. tests are run with setting 0, then with setting 1
        for sensitivity and equilibrium the basic switch settings remain untouched,
        i.e. the settings the modeler has set
        Switches are defined based on the name as there is no other property to identify them with

        :return: doc_df: Dataframe with doc information with type 'switch' added
        """
        for i, row in self.doc.iterrows():
            row = row.copy()
            if 'switch' in row['Py Name'].split('_') and row['type'] == 'constant':
                self.doc.at[i, 'type'] = 'switch'

    def replace_init_doc(self):
        """
        # this needs to go to the distance test
        Replaces init values like in the pysd routine also reads all list objects that are saved as string and returns
        them as lists

        needs to be run every time init values are needed from doc as doc is not saved

        Deprecated, only use the init replacement in the helper

        :return: updated doc
        """
        reg_list = ('DELAY1I', 'DELAY3I', 'SMOOTHI', 'SMOOTH3I')
        n_list = ('DELAY N', 'SMOOTH N')
        for i, row in self.doc.iterrows():
            if row['elements'] is not np.nan:
                row = row.copy()
                # there is a problem that if there is a function in the init of a delay function, it messes
                # up the elements completely
                if row['type'] == 'stock' or row['function'] in reg_list:
                    lst = row['elements']
                    if self.constant(lst[-1]):
                        lst[-1] = 'init %s' % row['Real Name']
                    self.doc.at[i, 'elements'] = lst
                elif row['function'] in n_list:
                    lst = row['elements']
                    if self.constant(lst[-2]):
                        lst[-2] = 'init %s' % row['Real Name']
                    self.doc.at[i, 'elements'] = lst
                else:
                    lst = row['elements']
                    self.doc.at[i, 'elements'] = lst

    def graph_coll(self):
        """
        Collects the info for making the model representation
        """
        for i, row in self.doc.iterrows():
            name = row['Real Name'].replace('"', '').replace('/', '').replace('*', '')
            if row['type'] in ['constant', 'switch', 'table function']:
                self.node_lst.append({'ID': i,
                                      'label': name,
                                      'image': os.path.join(self.folder_dict['graphviz'], 'no.png'),
                                      'labelloc': 't',
                                      'URL': '',
                                      'fixedsize': 'False',
                                      'width': '0',
                                      'height': '0',
                                      })
            elif row['type'] in ['stock', 'flow', 'auxiliary']:
                self.node_lst.append({'ID': i,
                                      'label': name,
                                      'image': '',
                                      'labelloc': 't',
                                      'URL': '',
                                      'fixedsize': 'True',
                                      'width': self.node_width,
                                      'height': self.node_height,
                                      })
                elements = row['elements']
                elements = set(elements)
                for el in elements:
                    if not self.constant(el):
                        if el not in ['Time', 'TIME STEP', 'INITIAL TIME', 'FINAL TIME', 'SAVEPER']:
                            if '=' not in el:
                                try:
                                    # if the models are not prepared with the helper, this is problematic with
                                    # delay functions with functions in the init statement
                                    self.edge_lst.append((self.doc.loc[self.doc['Real Name'] == el].index[0], i))
                                except IndexError:
                                    pass

    def collect_doc(self):
        """
        not all types are used at all times

        Order of output: Const, Builtin, Stocks, Endogenous, Flows, Switches

        flows and endo overlap, might be worth having a look
        stocks and endo overlap as well

        type dfs need to be used for input in tests as they contain the pynames and values

        :return: dataframes for different types of variables
        """
        # the variable type doc slices could be grouped in a dict to keep order
        self.const = self.doc.loc[self.doc['type'] == 'constant'].reset_index()
        self.builtin = self.doc.loc[self.doc['type'] == 'builtin'].reset_index()
        self.stocks = self.doc.loc[self.doc['type'] == 'stock'].reset_index()
        self.flows = self.doc.loc[self.doc['type'] == 'flow'].reset_index()
        self.switches = self.doc.loc[self.doc['type'] == 'switch'].reset_index()
        self.tables = self.doc.loc[self.doc['type'] == 'table function'].reset_index()
        # endo now doesn't include the flows anymore (be careful with use)
        # endo are now all endogenous variables except stock and flows
        self.endo = self.doc.loc[~self.doc['type'].isin(['constant', 'builtin', 'stock', 'flow', 'switch',
                                                         'table function', 'subscript list',
                                                         'subscripted constant'])].reset_index()

        self.base_unit = self.doc['Base Unit'][0]
        # the output names could also be gathered in a dict
        self.endo_names = self.endo['Real Name'].tolist()
        self.flow_names = self.flows['Real Name'].tolist()
        self.stock_names = self.stocks['Real Name'].tolist()
        # endo_names however includes all endogenous variables
        self.endo_names.extend(self.stock_names)
        self.endo_names.extend(self.flow_names)
        # exo_names needs to keep the order of the const DataFrame, list or pd.series are possible
        # currently pd.series is chosen but list would be prettier with the other lists (where order is not relevant)
        self.exo_names = self.const['Py Name']
        self.builtin_names = self.builtin['Py Name']
        self.graph_coll()

    def set_base_params(self):
        """
        is over written when equimode is used
        :return:
        """
        self.base_params = self.const['value']
        self.base_builtin = self.builtin['value']

    def create_base(self):
        """
        creates the base run with the output of all endogenous variables
        this is to read in the base values for exogenous variables
        :return:
        """
        self.base_full = self.model.run()

    def update_base(self):
        """
        updating the base is necessary for sensitivity runs in the equi mode to make sure
        that the base run is the equilibrium run
        """
        input_dict = dict(zip(self.exo_names, self.base_params))
        base = self.model.run(params=input_dict, return_columns=self.endo_names)
        self.base = Run('base', 'base')
        self.base.add_run(base)
        self.base.var_types()

    def read_exo_base(self):
        """
        reads in the values of all constants and builtins (vensim builtins),
        to have a starting point for test analysis

        :return: updated const and builtin dataframes
        """
        for i, row in self.const.iterrows():
            self.const.loc[i, 'value'] = self.base_full[row['Real Name']].iloc[0]
        for i, row in self.builtin.iterrows():
            self.builtin.loc[i, 'value'] = self.base_full[row['Real Name']].iloc[0]

    def save_csv(self, csv_name, df, test):
        """
        Saves data to .csv, is used for output files that will be rewritten for every iteration
        :param df: Dataframe with the data
        :param csv_name: string, name of the output file
        :param test: originating test (e.g. 'sens', 'equi')
        :return: saved .csv
        """
        # the replace elements are just to make sure that there are no naming error issues
        name = csv_name.replace('.py', '').replace('"', '').replace('/', '').replace('*', '')
        return df.to_csv(os.path.join(self.folder_dict[test], '%s.csv' % name), index=True, header=True)

    def append_csv(self, csv_name, df, test):
        """
        Same as save_csv(), but appends data to .csv files
        This is used mainly for tracking files, such as time files
        :param df: Dataframe with the data
        :param csv_name: string, name of the output file
        :param test: originating test (e.g. 'sens', 'equi')
        :return: appended .csv
        """
        # the replace elements are just to make sure that there are no naming error issues
        name = csv_name.replace('.py', '').replace('"', '').replace('/', '').replace('*', '')
        # this does the same as save, but adds to csv, can be used for time and error tracking
        if os.path.isfile(os.path.join(self.folder_dict[test], '%s.csv' % name)):
            with open(os.path.join(self.folder_dict[test], '%s.csv' % name), 'a') as f:
                df.to_csv(f, header=False)
        else:
            self.save_csv(name, df, test)

    def concat_csv(self, name, new_list, test):
        """
        concat together dataframes
        used for param settings

        :param name: str, name of file
        :param new_list: list, elements to be added
        :param test: str, save location
        """
        name = name.replace('.py', '').replace('"', '').replace('/', '').replace('*', '')
        new_df = pd.DataFrame(new_list)
        if os.path.isfile(os.path.join(self.folder_dict[test], '%s.csv' % name)):
            old_df = pd.read_csv(os.path.join(self.folder_dict[test], '%s.csv' % name), index_col=0)
        else:
            old_df = pd.DataFrame()
        con_df = pd.concat([old_df, new_df], axis=0)
        con_df.to_csv(os.path.join(self.folder_dict[test], '%s.csv' % name))

    def write_run_error_file(self, err_lst):
        """
        write run time errors to file

        :param err_lst: list, list of runtime errors collected from run with tracking
        """
        error_file = 'error_file.csv'
        err_df = pd.DataFrame(err_lst)
        err_df.columns = ['Source', 'Error Type', 'Description', 'Location', 'Parameters', 'Change to Base']
        err_df.loc[:, 'Location'] = err_df.loc[:, 'Location'].astype(str)
        err_df.loc[:, 'Description'] = err_df.loc[:, 'Description'].astype(str)
        err_df.drop_duplicates(subset=['Source', 'Error Type', 'Location'], inplace=True)
        if os.path.isfile(os.path.join(self.folder_dict['base'], error_file)):
            with open(os.path.join(self.folder_dict['base'], error_file), 'a') as f:
                err_df.to_csv(f, header=False)
        else:
            self.save_csv('error_file', err_df, 'base')

    def write_time_data(self):
        """
        reads time data with some additional information and adds it to the time list
        time list is used to track the times for each model and is saved at the end
        :return: time list, list of tuples
        """
        stats_df = pd.read_csv(os.path.join(self.folder_dict['doc'], '%s.csv' % self.out_name))
        time_info = [(self.out_name, self.elapsed / 60, stats_df.loc(axis=1)['FINAL TIME'][0] -
                      stats_df.loc(axis=1)['INITIAL TIME'][0], stats_df.loc(axis=1)['TIME STEP'][0],
                      len(stats_df.index))]
        self.save_lst_csv(time_info, '%s_time' % self.test, 'source', columns=['Model Name', 'Time for Test',
                                                                               'Time Horizon', 'Time Step',
                                                                               'Number of Vars'])

    def write_error_file(self, e):
        """
        this is to use scripting error files that originate from code
        :param e: incurred exception when executing code
        :return: reported error
        """
        f = open(os.path.join(self.folder_dict['source'], self.err_file_name), 'a')
        f.write('%s : %s\n' % (str(self.out_name), str(e)))
        f.close()

    def save_lst_csv(self, lst, csv_name, test, columns=None, append=True):
        """
        This does the same as save_csv or append_csv, but takes a list as input and transforms it to a Dataframe

        :param csv_name:
        :param lst: list with the data
        :param test: originating test (e.g. 'sens', 'equi')
        :param columns: optional, can add columns to the dataframe if necessary
        :param append: boolean, whether or not it should be appended
        :return: saved csv
        """
        df = pd.DataFrame(lst)
        if columns is not None:
            df.columns = columns
        if append:
            self.append_csv(csv_name, df, test)
        else:
            self.save_csv(csv_name, df, test)

    def check_names(self):
        """
        Test function to check if names are correct from MP, not needed anymore
        :return:
        """
        for run in self.run_lst:
            if run.name == run.check_name:
                print('same')
            else:
                print('not same')

    def iterate_endo_plots(self, key, full_df):
        """
        create the sensitivity graphs with all sensitivity runs for each endogenous variable

        :param key:
        :param full_df: df, complete with all runs to be plotted
        :return: plots to be created (with plot function)
        """

        for i, var in enumerate(self.endo_names):
            # sensitivity percentage is only relevant for sensitivity graphs

            name = '%s_%s' % (var, key)
            unit = self.doc.loc[self.doc['Real Name'] == var]['Unit'].values
            endo_run = full_df.loc(axis=1)[:, var]
            endo_run.columns = endo_run.columns.droplevel(1)
            self.save_endo_plots(endo_run, unit, name)
            plt.close('all')

    def save_endo_plots(self, endo_run, unit, name):
        """
        saves the endo plots via the saving pipe
        :param endo_run: pd.Dataframe(), run results
        :param unit: str, unit to be added to the y-axis if not sens or norm
        :param name: name of the graph file
        """
        # opening pipes all over the place might be not the best idea, one pipe for all saving might be better
        pipe = Plass(self)
        type_name = self.test
        if self.testing_mode:
            pipe.save_csv(endo_run, type_name, name)
        pipe.create_sens_plot(endo_run, unit, name, type_name)

    def initialize_test(self):
        """
        This function exists because it's overwritten in the equilibrium because there are
        additional elements to initialize

        :return:
        """
        self.initialize_base()

    def initialize_base(self):
        """
        general initialization of the test, is run for every __mp_main__
        """
        # start the timer
        self.start_test()
        # start the error and full_df tracking in the model
        self.model.set_tracking()
        # create the folder dict (could be moved to init)
        self.make_folder_dict()
        # read the doc file
        self.read_doc_file()
        # this should not be necessary, switches are identified when the doc file is read in 26.07.18/sk
        # self.identify_switches()
        # this should not be used anymore, init replacement is in the helper
        # self.replace_init_doc()
        # gather the dfs for vartypes and output lists
        self.collect_doc()
        # we need to create the base first to read in the base params from the base run and update the doc file
        self.create_base()
        self.read_exo_base()
        self.set_base_params()
        # test with udpate base at the end, has to be after set base params, otherwise the sensitivity base run is not
        # equi 21.06.18/sk
        self.update_base()

    def prepare_test(self):
        """
        Place holder for all tests for preparation
        """
        pass

    def run_test_mp(self, run=None):
        """
        function for running the tests
        :param run: run object, with settings and output defined
        :return: pd.Dataframe(), to be added to thte result list
        """
        if run is not None:
            args = run, self.flow_names, self.stock_names, self.test_name
            res = self.model.run_with_tracking(args)
            return res

    def run_test_helper(self):
        """
        helper for non MP execution, does the iteration over the run list
        :return: result list to be collected
        """
        res_lst = []
        for run in self.run_lst:
            args = run, self.flow_names, self.stock_names, self.test_name
            res = self.model.run_with_tracking(args)
            res_lst.append(res)
        return res_lst

    def run_test(self):
        """
        placeholder for run test (which is when MP setting is off for the test)

        only Equilibrium uses this functionality at the moment
        """
        pass

    def collect_res(self, res):
        """
        collects the results from the test execution and prepares them for further use


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
                    self.model.add_run(run.run, run.name, run.full_ID)
                    if self.test in self.test_with_sens:
                        run.treat_run(self.base.run)
                    else:
                        run.var_types()
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

    def save_output(self):
        """
        placeholder for saving output which is individual for every test
        """
        pass

    def save_full_output_mp(self, key):
        """
        used by ext

        :param key:
        :return:
        """
        full_df = pd.read_csv(os.path.join(self.folder_dict[self.test], 'full_df_%s.csv' % key), index_col=0,
                              header=[0, 1], dtype=np.float64)
        self.iterate_endo_plots(key, full_df)
        # opening pipes all over the place might be not the best idea, one pipe for all saving might be better
        pipe = Plass(self)
        pipe.create_model(key, full_df, self.test)

    def save_full_output_helper(self):
        """
        helper function for when it's not run by MP
        does the iteration sequentially

        """
        for key in self.model.full_df_dict.keys():
            self.save_full_output_mp(key)

    def save_ind_output_mp(self, run):
        """
        placeholder for individual output saving
        only used in sensitivity

        :param run: run object with all the settings
        """
        pass

    def save_ind_output_helper(self):
        """
        helper function for when it's not run by MP
        """

        for run in self.run_lst:
            self.save_ind_output_mp(run)

    def start_test(self):
        """
        starting the timer for performance measurement
        """
        self.start = timer()

    def end_test(self):
        """
        this is house cleaning that has to be done for every test

        """

        self.model.check_err_list(self.test_name)
        self.write_run_error_file(self.model.err_lst)
        if self.testing_mode:
            for run in self.run_lst:
                self.run_coll.append(run.input_dict)
            self.concat_csv('param_settings', self.run_coll, 'base')
        self.end = timer()
        self.elapsed = (self.end - self.start) / 60
        self.write_time_data()

    def write_report(self):
        """
        placeholder for report writing which is individual for each test
        """
        pass
