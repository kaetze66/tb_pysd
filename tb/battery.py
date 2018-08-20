"""
the battery.py document contains the battery class which sets up the test battery and runs the different components
and the pipe class which handles the tracking and support for MP runs

maybe it would make sense to put the folder dict here and pass it on to make it more consistent
currently it's in basetest and savingpipe, which is a potential cause for errors if they are not the same


battery class has all the methods for running the entire battery of tests

can be run single thread and MP

MP will produce compatibility issues, but is a lot faster

battery has PySD helper now included
battery mainly does:

- set up of test list
- writing of pipe line files (for MP) or handling of pipe line (for non MP)
- determining the MP settings


Version 0.3
Update 30.07.18/sk
"""

import os
import pandas as pd
import shutil
import datetime
import pickle
import subprocess
from configparser import ConfigParser
from tb import descriptives as desc
from tb.tests import Sensitivity, MonteCarlo, Equilibrium, TimeStep, Switches, Distance, KnockOut, Extreme, Horizon
from tb.tb_backend.builder import Batch, ExecFile
from tb.tb_backend.pysdhelper import PysdHelper
from tb.tb_backend.report import Report


class Battery:
    """
    Battery handles the setting up and execution of the tests

    """
    def __init__(self, folder, mp_setting=True, first=None, last=None, distance=False, knockout=False):
        # total elapsed is currently broken, need to be added again
        self.total_elapsed = 0
        # test list is initialized for adding the tests later on
        self.test_lst = []
        self.file_lst = []

        # these are the settings that come from the testingbattery master file
        self.folder = folder
        self.first_file = first
        self.last_file = last
        self.mp_setting = mp_setting
        # distance is not in the config file because it's currently useless
        self.distance = distance
        # knockout is not in the config file because it's currently wrong 18.06.18/sk
        self.knockout = knockout

        # defining the report and debug folder because folder dicts are limited to tests (defined in base test)
        # could also be passed from here, but well...
        self.report_folder = os.path.join(os.path.split(self.folder)[0], 'report', os.path.split(self.folder)[1])
        self.debug_folder = os.path.join(self.report_folder, '_debug', '_runtime')
        # config file is defined and read in here
        self.cf = ConfigParser()
        # config folder doesn't change, so it's put here to it's on one line
        self.cf.read(os.path.join(os.path.split(folder)[0], '_config', 'tb_config.ini'))

        # reading in the values from the config file for test parameters
        self.sensitivity_percentage = self.cf['test parameters'].getfloat('sensitivity_percentage', fallback=0.1)
        self.montecarlo_percentage = self.cf['test parameters'].getfloat('montecarlo_percentage', fallback=0.5)
        self.montecarlo_runs = self.cf['test parameters'].getint('montecarlo_runs', fallback=100)
        self.equilibrium_method = self.cf['test parameters'].getint('equilibrium_method', fallback=1)
        self.equilibrium_increment = self.cf['test parameters'].getfloat('equilibrium_increment', fallback=0.2)
        self.equilibrium_incset = self.cf['test parameters'].getboolean('equilibrium_incset', fallback=True)
        self.extreme_max = self.cf['test parameters'].getint('extreme_max', fallback=10)
        # reading in the setting for the pysd helper from the config file
        self.pysdhelper_setting = self.cf['component control'].getboolean('PySD_helper', fallback=True)
        # getting the translation setting for the initialization
        self.set_trans = self.cf['component control'].getboolean('translation', fallback=True)
        # getting the mode as well, is needed for the clean up of some .csv files
        self.testing_mode = self.cf['testing'].getboolean('testing_mode')

        # define the MP settings, if MP is true, then it is overwritten
        self.cores = 'no MP'
        self.processes = 'no MP'
        self.max_tasks = 'no MP'

    # initialization methods #

    def clear_reports(self):
        """
        clears out the reports (html files) from the source folder

        :return:
        """
        rep_lst = [f for f in os.listdir(self.folder) if f.endswith('.html')]
        for file in rep_lst:
            os.remove(os.path.join(self.folder, file))

    def run_report(self):
        """
        Run report is used when the PySD helper is on to extract statistics from the original file,
        prior to the pysd helper adjustments

        the method is in descriptives.py and should be rewritten at some point
        :return:
        """
        desc.init(self.folder)
        trans_elapsed = desc.create_report(self.first_file, self.last_file)
        self.total_elapsed += trans_elapsed

    def run_pysdhelper(self):
        """
        run pysd helper is the pysd helper integration and runs it on all models that are in the testing folder
        previously treated models are not rerun again

        :return:
        """
        file_lst = [f for f in os.listdir(self.folder) if f.endswith('.mdl')]
        # this could potentially lead to the result that a model that has the original file name ending with treated,
        # not being treated, but then just rename and it works
        file_lst = [f for f in file_lst if not f.endswith('_treated.mdl')]

        for file in file_lst:
            model = PysdHelper(self.folder, file)
            model.run_helper()

    def clear_error_files(self):
        """
        error files are only removed when a new model is translated, presumably after new model changes have been done
        this allows running tests independently on the same translated version while keeping the errors

        :return:
        """
        file_lst = [f for f in os.listdir(self.folder) if f.endswith('.mdl')]
        for file in file_lst:
            try:
                os.remove(os.path.join(self.folder, file.rsplit('.', 1)[0], 'error_file.csv'))
            except FileNotFoundError:
                pass

    def run_translate(self):
        """
        run translate calls the full translate method from the descriptives.py and is run after the pysd helper

        :return:
        """
        if not self.pysdhelper_setting:
            desc.init(self.folder)
        trans_elapsed = desc.full_translate(self.first_file, self.last_file)
        # currently the time elapsed is broken
        self.total_elapsed += trans_elapsed

    def load_files(self):
        """
        load files gets the file list after the translation for the tests
        needs to be expanded to xmile types when xmile translation is done

        :return: list of files
        """
        # first we load the .mdl files then rename it to py because if there are macros,
        # pysd will create .py files in the same folder which creates problems in the testing of the models
        self.file_lst = [f for f in os.listdir(self.folder) if f.endswith('.mdl')]
        self.file_lst = [f.replace('.mdl', '.py') for f in self.file_lst]

    def clear_base(self):
        """
        Removes the reporting and result files in the base folder (each models individual results folder)

        :return:
        """
        for file in self.file_lst:
            try:
                olfiles = [f for f in os.listdir(os.path.join(self.folder, file.rsplit('.', 1)[0])) if
                           f.endswith('.csv')]
            except FileNotFoundError:
                olfiles = []
            if olfiles:
                for olfile in olfiles:
                    # error file needs to be kept because it might have errors from translation in it
                    if not olfile == 'error_file.csv':
                        os.remove(os.path.join(self.folder, file.rsplit('.', 1)[0], olfile))

    def init_reports(self):
        """
        initializes the report page
        since name of the model changes if pysd helper is active, the information is saved in different folders
        to make sure it all drops to the same model, the statistics, the psyd helper actions and the working model
        are added to the report based on pickled report tuples

        :return:
        """
        # grabbing the pickle file list from the report folder
        pfile_lst = [f for f in os.listdir(self.report_folder) if f.endswith('.pickle')]
        for file in self.file_lst:
            print('Creating report for', file)
            report = Report(self.folder, file)
            # setting the styles to make sure the tables all look the same
            report.set_styles()
            report.write_quicklinks()
            # pickle files that are for this model are selected here
            rep_lst = [f for f in pfile_lst if f.startswith(file.replace('_treated', '').rsplit('.', 1)[0])]
            for rep in rep_lst:
                # if there are 3 pickle files, that means that the pysd helper has been run and thus it makes sense to
                # report the original model, if not, it doesn't make sense because it's the same as the working model
                if len(rep_lst) == 3:
                    if rep.endswith('orig.pickle'):
                        pickle_in = open(os.path.join(self.report_folder, rep), 'rb')
                        rep_tpl = pickle.load(pickle_in)
                        report.write_trans(rep_tpl)
                if rep.endswith('helper.pickle'):
                    pickle_in = open(os.path.join(self.report_folder, rep), 'rb')
                    rep_tpl = pickle.load(pickle_in)
                    report.write_helper(rep_tpl)
                # an argument could be made that the working model doesn't need to be reported when the original model
                # is reported, since the changes likely are marginal in terms of numbers, but it's not that much to do,
                # so we'll just leave it
                if rep.endswith('work.pickle'):
                    pickle_in = open(os.path.join(self.report_folder, rep), 'rb')
                    rep_tpl = pickle.load(pickle_in)
                    report.write_trans(rep_tpl)
            report.save_report()

    def set_mp_settings(self):
        """
        determines the number of CPUs used and creates the settings for the test execution

        is only run if the mp setting is on

        values are set to 'No MP' unless this method is run

        :return:
        """
        # getting the MP settings regardless of MP setting
        cpu_cnt = os.cpu_count()
        # 0.69 is set to keep some cpu power for regular operation but still using as much power as possible for
        # calculations
        mp_cpu = round(cpu_cnt * 0.69)

        self.cores = mp_cpu
        # processes could probably be higher, but for now lets keep it at cpu number
        self.processes = mp_cpu
        # max tasks defines the limit until a child is relaunched
        # value is set pretty randomly, if RAM issues are reported, this needs to be lowered
        self.max_tasks = 1000

    def prep_test_lst(self):
        """
        component control, true means test is run
        setting comes from config file, defaults to True
        :return:
        """

        equilibrium = self.cf['component control'].getboolean('equilibrium', fallback=True)
        sensitivity = self.cf['component control'].getboolean('sensitivity', fallback=True)
        switches = self.cf['component control'].getboolean('switches', fallback=True)
        timestep = self.cf['component control'].getboolean('timestep', fallback=True)
        montecarlo = self.cf['component control'].getboolean('montecarlo', fallback=True)
        extreme = self.cf['component control'].getboolean('extreme', fallback=True)
        horizon = self.cf['component control'].getboolean('horizon', fallback=True)
        for mdl_file in self.file_lst[self.first_file:self.last_file]:
            err_code = 'Translation Error'
            try:
                # this has to be in the right order for the testing sequence
                # if run single thread, the list is going to determine order
                # if run with MP, then the tests are run in alphabetical order, thus need to have the test ID
                # 02 - 09 are kept open for other test that might be linked to later tests
                # 10 - 19 are sensitivity tests
                # 20+ are other tests
                # 99 is used by the dummy test generator
                if equilibrium:
                    err_code = 'Translation Error equi'
                    test = Equilibrium(self.folder, mdl_file, self.equilibrium_method, self.equilibrium_increment,
                                       self.equilibrium_incset)
                    test.testID = '00'
                    self.test_lst.append(test)
                # distance is currently useless, but might be useful for other tests at some point
                # distance is currently just handled in testing mode and the setting comes from the testing battery
                # contrary to all other tests which get the settings from the config file
                if self.distance:
                    err_code = 'Translation Error dist'
                    test = Distance(self.folder, mdl_file)
                    test.testID = '01'
                    self.test_lst.append(test)
                if montecarlo:
                    err_code = 'Translation Error mc'
                    test = MonteCarlo(self.folder, mdl_file, self.montecarlo_percentage, self.montecarlo_runs)
                    test.testID = '10'
                    self.test_lst.append(test)
                if sensitivity:
                    err_code = 'Translation Error sens'
                    test = Sensitivity(self.folder, mdl_file, self.sensitivity_percentage)
                    test.testID = '11'
                    self.test_lst.append(test)
                if extreme:
                    err_code = 'Translation Error ext'
                    test = Extreme(self.folder, mdl_file, self.extreme_max)
                    test.testID = '20'
                    self.test_lst.append(test)
                if timestep:
                    err_code = 'Translation Error ts'
                    test = TimeStep(self.folder, mdl_file)
                    test.testID = '21'
                    self.test_lst.append(test)
                if self.knockout:
                    err_code = 'Translation Error ko'
                    test = KnockOut(self.folder, mdl_file)
                    test.testID = '22'
                    self.test_lst.append(test)
                if horizon:
                    err_code = 'Translation Error hori'
                    test = Horizon(self.folder, mdl_file)
                    test.testID = '23'
                    self.test_lst.append(test)
                if switches:
                    err_code = 'Translation Error swit'
                    test = Switches(self.folder, mdl_file)
                    test.testID = '24'
                    self.test_lst.append(test)
            except Exception as e:
                # there should be no errors here but better safe than sorry
                f = open(os.path.join(self.folder, 'exec_error_file.txt'), 'a')
                f.write('%s, %s : %s\n' % (err_code, str(mdl_file), str(e)))
                f.close()

    def create_batch_file(self):
        """
        creates the batch file for mp file execution
        batch file contents are in the builder

        :return:
        """
        batch = Batch(self.folder)
        batch.write_batch()

    # run pipe methods #

    @staticmethod
    def clean_files(folder, ftype):
        """
        cleans all files in a folder defined by file extension

        :param folder: folder to be cleaned
        :param ftype: file type (file extension) to be deleted
        :return:
        """
        file_lst = [f for f in os.listdir(folder) if f.endswith(ftype)]
        for f in file_lst:
            os.remove(os.path.join(folder, f))

    def report_errors(self):
        """
        report files are handled after all tests are executed and added to the report

        :return:
        """

        for mdl_file in self.file_lst[self.first_file:self.last_file]:
            rep = Report(self.folder, mdl_file)

            error_link = os.path.join(self.folder, mdl_file.rsplit('.', 1)[0], 'error_file.csv')
            try:
                error_df = pd.read_csv(error_link, index_col=0)
                error_df = error_df.loc[error_df['Error Type'] != 'No Error']
                cnts = error_df['Source'].value_counts()
                # error link needs relative path to make sure results can be copied to other locations
                error_link = error_link.replace(self.folder, '')
            except FileNotFoundError:
                # if the reading of the error file fails, we pass an emtpy pandas series to the report tuple
                cnts = pd.Series()
                # also the error link is None, this triggers that errors are not reported in the report thing
                error_link = None
            rep_tpl = ('Errors', cnts, error_link)
            rep.write_errors(rep_tpl)
            rep.save_report()

    def initialize_battery(self):
        """
        Runs all pre test cleaning, translation, loading of files and creation of test list


        :return:
        """
        # html report writing is appending code to the file, so we need to clear them first before we do anything
        self.clear_reports()
        # translation also removes the time DB, clears the report folder, etc.
        # when the descriptives file is rewritten, then this stuff should be moved here
        # however, the logic of just cleaning out when a model is newly translated should be maintained
        if self.set_trans:
            if self.pysdhelper_setting:
                # if pysd helper is on, then statistics should be gathered first
                self.run_report()
                self.run_pysdhelper()
            self.clear_error_files()
            self.run_translate()
        # previous methods load their own files, now the file list is held steady
        self.load_files()
        # clears the base folder if translation is on
        if self.set_trans:
            self.clear_base()
        # with translation done the report pages are initialized
        self.init_reports()
        # mp settings are defined automatically to avoid computer overload, only works on windows
        if self.mp_setting:
            self.set_mp_settings()
        # define the test list based on settings
        self.prep_test_lst()
        # create the batch file that runs the testing if mp is on
        if self.mp_setting:
            self.create_batch_file()
        # due to mp, the counts need to be pickled
        # also it might be better to call the first category tests to avoid confusion,
        # something to think about 14.06.18/sk
        counts = {'models': 0, 'errors': 0, 'total': len(self.test_lst)}
        pickle_out = open(os.path.join(self.folder, 'counts.pickle'), 'wb')
        pickle.dump(counts, pickle_out)
        pickle_out.close()
        # need to return the length of the test list to tell the testingbattery.py file if the pipe needs to be launched
        return len(self.test_lst)

    def create_exec_files(self):
        """
        the exec files are the python files dropped in the run pipe folder
        the template for the exec files is in the builder
        they are only used if mp is set to true

        :return:
        """
        if self.mp_setting:
            for test in self.test_lst:
                file = ExecFile(self.folder, test)
                file.write_exec_file(self.processes, self.max_tasks, self.cores)

    def run_pipe(self):
        """
        running the pipe executes the tests

        if mp setting is on:

        - it launches the batch file
        - waits until batch file completes the work
        - cleans up and reports errors

        if mp setting is not on:

        - it takes the test list
        - runs each test sequentially
        - cleans up and reports errors

        :return:
        """
        if self.mp_setting:
            folder = os.path.split(self.folder)[0]
            subp = subprocess.Popen(os.path.join(folder, 'run_pipe_exec.bat'))
            subp.communicate()
            # subp.wait will wait for the run pipe to finish to continue
            subp.wait()
        else:
            # pipe is technically not necessary here because reporting could be handled in the battery itself
            # however, it makes sense to have functions if possible just in one place, so for clean code,
            # all reporting is moved to the pipe class
            pipe = Pipe(self.folder, self.cores, self.processes, self.max_tasks)
            # if error reporting is True, then the execution is not halted on an error
            # and the error is tracked in the exec_error_file
            error_reporting = True
            for test in self.test_lst:
                if error_reporting:
                    try:
                        # the structure for test execution has to stay the same for all tests
                        # exec file writes a helper function with this structure
                        test.initialize_test()
                        test.initiate_folder()
                        test.prepare_test()
                        if test.MP:
                            # helper functions take on the iteration through the running of tests that the mp would do
                            # returning the result and collecting it wouldn't be necessary here
                            # but structure needs to be kept
                            res = test.run_test_helper()
                            test.collect_res(res)
                        else:
                            test.run_test()
                        # helper functions take on the iteration through the saving that the mp would do
                        test.save_full_output_helper()
                        test.save_ind_output_helper()
                        test.save_output()
                        test.write_report()
                    except Exception as e:
                        # only execution errors are tracked here, not runtime errors
                        pipe.track_errors(test, e)
                else:
                    # this is just to test non MP test execution
                    # the structure for test execution has to stay the same for all tests
                    # exec file writes a helper function with this structure
                    test.initialize_test()
                    test.initiate_folder()
                    test.prepare_test()
                    if test.MP:
                        # helper functions take on the iteration through the running of tests that the mp would do
                        # returning the result and collecting it wouldn't be necessary here
                        # but structure needs to be kept
                        res = test.run_test_helper()
                        test.collect_res(res)
                    else:
                        test.run_test()
                    # helper functions take on the iteration through the saving that the mp would do
                    test.save_full_output_helper()
                    test.save_ind_output_helper()
                    test.save_output()
                    test.write_report()
                test.end_test()
                pipe.add_track(test)
                print('Model:', test.out_name, 'Test:', test.test_name)
        # any battery wide activities are placed here
        # works for both the MP and non MP execution
        self.clean_files(self.report_folder, '.pickle')
        self.report_errors()
        self.clean_files(self.folder, '.pickle')
        if not self.testing_mode:
            self.clean_files(self.folder, '.csv')


class Pipe:
    """
    Pipe is the collection of battery functions that MP execution needs to relaunch for functioning
    it's used to make sure that things go in the right place
    also it's more lightweight than the whole battery with all the settings
    only tracking functions are in here

    this way we also avoid overwriting the battery over and over again
    battery is just run once

    """

    def __init__(self, folder, cores, processes, maxtasks):
        self.folder = folder
        self.report_folder = os.path.join(os.path.split(self.folder)[0], 'report', os.path.split(self.folder)[1])
        self.debug_folder = os.path.join(self.report_folder, '_debug', '_runtime')
        self.cores = cores
        self.processes = processes
        self.max_tasks = maxtasks

    def track_errors(self, test, e):
        """
        writes the error to the exec error file
        currently the info provided is not that useful, more error text should be reported 14.06.18/sk


        :param test: test causing the execution error
        :param e: exception description
        :return:
        """
        # these are the execution errors, not runtime errors (they are in models)
        f = open(os.path.join(self.folder, 'exec_error_file.txt'), 'a')
        f.write('%s, %s : %s\n' % (str(test.test_name), str(test.out_name), str(e)))
        f.close()
        # the model causing the error is moved to the debug folder for troubleshooting
        shutil.copy2(test.file, self.debug_folder)
        # the counts have to be pickled because with mp, the pipe cannot keep track of it
        # (since it's relaunched with ever mp_main process)
        pickle_in = open(os.path.join(self.folder, 'counts.pickle'), 'rb')
        counts = pickle.load(pickle_in)
        counts['errors'] += 1
        pickle_out = open(os.path.join(self.folder, 'counts.pickle'), 'wb')
        pickle.dump(counts, pickle_out)
        pickle_out.close()

    def add_track(self, test):
        """
        writes the time data into the time DB
        this describes the time for the entire execution of the test, not just code run through
        (as in the individual time files)

        :param test: test being tracked
        :return:
        """
        date = datetime.datetime.now().date()
        time = datetime.datetime.now().time()
        inst = [(date, time, test.out_name, test.test_name, test.elapsed, self.cores, self.processes)]
        # making this a DF is probably overkill, should be simplified 14.06.18/sk
        time_df = pd.DataFrame(inst)
        # time data is always appended while the folder is not cleaned in translation
        if os.path.isfile(os.path.join(self.folder, 'TimeDB.csv')):
            with open(os.path.join(self.folder, 'TimeDB.csv'), 'a') as f:
                time_df.to_csv(f, header=False)
        else:
            # if file doesn't exist, new headings need to be provided
            time_df.columns = ['Date', 'Time', 'Model Name', 'Test Name', 'Time Elapsed', '# of cores',
                               '# of processes']
            time_df.to_csv(os.path.join(self.folder, 'TimeDB.csv'), index=True, header=True)
        # counts need to be pickled, since in mp pipe cannot keep track of them
        # (pipe is relaunched with every mp_main process)
        try:
            # exception handling is necessary when dummy tests are run
            pickle_in = open(os.path.join(self.folder, 'counts.pickle'), 'rb')
            counts = pickle.load(pickle_in)
            counts['models'] += 1
            # reporting the progress to make sure that pipe is running and hasn't killed itself
            print(counts['models'], 'tests done out of', counts['total'], 'with', counts['errors'], 'errors')
            pickle_out = open(os.path.join(self.folder, 'counts.pickle'), 'wb')
            pickle.dump(counts, pickle_out)
            pickle_out.close()
        except FileNotFoundError:
            pass
