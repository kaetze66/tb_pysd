"""
builder.py

Class to build the batch file and the run exec files

compatibility issues with:
- Apple and Linux systems
- if python is not installed (when testing battery is an executable, this needs to be adjusted)

moved prepare test into __main__ 19.06.18/sk

Version 0.2
Update 30.07.18/sk
"""

import sys
import os
import textwrap


class Batch:
    """
    Class to build the batch file for the run pipe

    the reason why the tests have to be launched with a batch file is that the the cached version of
    pysd screws up the multiprocessing in the execution

    error message is that it can't find the model (which is the cached model)

    all tests are compatible with the batch file now 05.06.18/sk
    """

    def __init__(self, folder):
        # for this to be run, python needs to be installed, maybe add a check for this in the battery
        # this is probably not necessary anmyore 05.06.18/sk
        # self.python_loc = sys.executable
        self.folder = os.path.split(folder)[0]
        self.bat_name = 'run_pipe_exec.bat'

    def build_batch(self):
        """
        this sets up the elements of the batch file

        it is saved in the root, in the testingbattery
        :return:
        """
        self.bat_head = '''
                    @echo off
                    cd run_pipe
                    set cnt=0
                    set run=0
                    for %%a in (.\*) do set /a cnt+=1
                    echo File count = %cnt%
                    :Loop
                    for %%f in (.\*) do (
                     set filename=%%f
                     echo %filename%
                     goto RunFile)
                    :RunFile
                    '''
        self.bat_mid = '%s ' % sys.executable
        self.bat_foot = '''%filename% %*
                    echo batch done
                    set /a run+=1
                    echo Run count = %run%
                    del %filename%
                    if %cnt% gtr %run% goto Loop
                    exit 0'''

    def clear_batch(self):
        """
        clears the batch file if it exists
        :return:
        """
        if os.path.isfile(os.path.join(self.folder, self.bat_name)):
            os.remove(os.path.join(self.folder, self.bat_name))

    def write_batch(self):
        """
        writes and saves the batch to execute the runpipe
        :return:
        """
        self.clear_batch()
        self.build_batch()
        with open(os.path.join(self.folder, self.bat_name), 'a') as f:
            f.write(textwrap.dedent(self.bat_head + self.bat_mid + self.bat_foot))


class ExecFile:
    """
    Template to produce test files for the run pipe
    each file is one test and is launched through the batch file
    this is not necessary if MP is not set to true

    """

    def __init__(self, folder, test):
        self.folder = folder
        self.test = test

    def build_test_string(self):
        """
        builds the test string (the initialization of the test)

        new tests need to be added here with the information that is necessary

        works for all tests 05.06.18/sk

        :return:
        """
        if self.test.test_name == 'sensitivity':
            self.test_string = '''test = %s(folder,'%s',%s)''' % (
                self.test.class_name, self.test.file_name, self.test.sp)
        elif self.test.test_name == 'montecarlo':
            self.test_string = '''test = %s(folder,'%s',%s,%s)''' % (
                self.test.class_name, self.test.file_name, self.test.sp, self.test.nmb_runs)
        elif self.test.test_name == 'equilibrium':
            self.test_string = '''test = %s(folder,'%s',%s,%s,%s)''' % (
                self.test.class_name, self.test.file_name, self.test.equi_method, self.test.sp, self.test.set_inc)
        elif self.test.test_name == 'timestep':
            self.test_string = '''test = %s(folder,'%s')''' % (self.test.class_name, self.test.file_name)
        elif self.test.test_name == 'switches':
            self.test_string = '''test = %s(folder,'%s')''' % (self.test.class_name, self.test.file_name)
        elif self.test.test_name == 'distance':
            self.test_string = '''test = %s(folder,'%s')''' % (self.test.class_name, self.test.file_name)
        elif self.test.test_name == 'knockout':
            self.test_string = '''test = %s(folder,'%s')''' % (self.test.class_name, self.test.file_name)
        elif self.test.test_name == 'extreme':
            self.test_string = '''test = %s(folder,'%s',%s)''' % (
                self.test.class_name, self.test.file_name, self.test.max)
        elif self.test.test_name == 'horizon':
            self.test_string = '''test = %s(folder,'%s')''' % (self.test.class_name, self.test.file_name)

    def set_import(self, processes, max_tasks, cores):
        """
        creates the script for the run pipe

        currently both battery and test instances are recreated in the script, which is not ideal because that doubles
        the workload, both battery should be the same as the ones creating the script

        new tests need to be added to the import list

        :param processes: int, determines the number of parallel processes
        :param max_tasks: int, determines how many tasks before a pipe is renewed
        :param cores: int, determines how many cores are used for calculation
        :return:
        """
        self.imp_str = '''
            import sys
            sys.path.append(r'%s')
            import pathos.pools as pp
            from tb.tests import Sensitivity, MonteCarlo, Equilibrium, TimeStep, Switches
            from tb.tests import Distance, KnockOut, Extreme, Horizon
            from tb.battery import Pipe
            def run_mp(input):
                res = test.run_test_mp(input)
                return res
            def plot_ind_mp(input):
                test.save_ind_output_mp(input)
            def plot_full_mp(input):
                test.save_full_output_mp(input)
            def exec_pipe(test):
                test.initialize_test()
                if __name__ == '__main__':
                    test.initiate_folder()
                    test.prepare_test()
                    pool = pp.ProcessPool(nodes=%s)
                    if test.MP:
                        # different mapping methods should be checked for speed
                        res = pool.map(run_mp, [run for run in test.run_lst])
                        pool.close()
                        pool.join()
                        test.collect_res(res)
                        pool.restart()
                    else:
                        test.run_test()
                    print('Starting full output')
                    # different mapping methods should be checked for speed
                    pool.map(plot_full_mp,[key for key in test.model.full_df_dict])
                    pool.close()
                    pool.join()
                    pool.restart()
                    print('Starting ind output')
                    # different mapping methods should be checked for speed
                    pool.map(plot_ind_mp, [run for run in test.run_lst])
                    pool.close()
                    pool.join()
                    test.save_output()
                    test.write_report()
            folder = r'%s'
            pipe = Pipe(folder,%s,%s,%s)
            %s
            try:
                exec_pipe(test)
            except Exception as e:
                pipe.track_errors(test,e)
            print(__name__)
            print('python done')
            if __name__ == '__main__':
                test.end_test()
                pipe.add_track(test)
                print('Model:', test.out_name, 'Test:', test.test_name)
            ''' % (os.path.split(self.folder)[0],
                   processes,
                   self.folder,
                   cores, processes, max_tasks,
                   self.test_string)

    def write_exec_file(self, processes, max_tasks, cores):
        """
        summarizes the writing of exec files

        parameters are just to pass through to set import

        :param processes: int, determines the number of parallel processes
        :param max_tasks: int, determines how many tasks before a pipe is renewed
        :param cores: int, determines how many cores are used for calculation
        :return:
        """
        self.build_test_string()
        self.set_import(processes, max_tasks, cores)
        base_folder = os.path.split(self.folder)[0]
        with open(os.path.join(base_folder, 'run_pipe',
                               '%s_%s%s.py' % (self.test.out_name, self.test.testID, self.test.test_name)), 'a') as f:
            f.write(textwrap.dedent(self.imp_str))
