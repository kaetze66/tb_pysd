"""
this code regulates the sequential handling of the testing sequence and manages all elements for the
entire testing process

version 0.1
02.03.18/sk
"""

from tb import sensitivity as sens
from tb import equilibrium as equi
from tb import descriptives as desc
from timeit import default_timer as timer
from tb.tb_backend import fileops as fops

# component control, true means test is run
translation = True
equilibrium = True
sensitivity = True
# file settings: None means all files in that direction,
# i.e. None in first file is starting index 0, None in last file is ending index -1
first_file = None
last_file = None

# initiating the time for total tests
total_elapsed = 0

#general settings for testing
current_testing_dir = 'test'
folder = './%s' % current_testing_dir

if translation:
    # descriptives is set up differently because much of the info stays in that file and splitting it up is too
    # laborious at the moment, maybe something to be done in the future, but currently it works
    desc.init(folder)
    # only the time is reported back, every other output is saved in files and reused afterwards
    # from those files (e.g. doc file)
    trans_elapsed = desc.full_translate(first_file,last_file)
    total_elapsed += trans_elapsed


if equilibrium:
    # setting up the configs for the equilibrium test
    equilibrium_percentage, equi_method, equi_model_count, equi_time_lst, equi_error_file, equi_error_cnt = \
        equi.config(equilibrium_percentage=0.1,equi_method=1)
    # launch the init settings for the equilibrium test
    equi.init(folder,equilibrium_percentage)
    # launch the init for the fileops folder, technically not necessary if it has been done in a previous test,
    # but since it's just setting a couple of params, it is not that relevant
    fops.init(folder)

    equi_total_start = timer()

    # File Management
    mdl_files = fops.load_files('py')
    # possibly the file management should go outside of the test, iterating tests over one model
    # instead of iterating models over one test
    # might prove easier for multiple processing

    for model in mdl_files[first_file:last_file]:
        print(model)
        model_name = model.split('.')[0]
        # individual models are timed as well, result is saved in the time_df for this test
        equi_ind_start = timer()
        equi_model_count += 1
        try:
            err_lst = equi.equilibrium(model,equi_method=1,incremental=True)
        except Exception as e:
            fops.write_error_file(model_name,e,equi_error_file)
            # if an error happens in the file execution, then the runtime error list is empty and needs to be
            # created and passed as empty
            err_lst = []
            # this goes into the runtime error file to make sure we know that the test script has an error
            err_lst.append(('Incomplete','Incomplete','Incomplete','Incomplete','Incomplete','Incomplete'))
            fops.move_mdl_debug(model,'run_debug')
        #print('Models: ', equi_model_count)
        equi_ind_end = timer()
        # this saves the runtime errors to file
        fops.write_run_error_file(err_lst, 'error_file.csv')
        equi_time_lst = fops.read_time_data(model_name, equi_ind_end - equi_ind_start, equi_time_lst)
    fops.save_lst_csv(equi_time_lst, 'equi_time', 'source', columns=['Model Name', 'Time Equi',
                                                                   'Time Horizon', 'Time Step', 'Number of Vars'])

    # calculating the elapsed time for the test and adding it to the total
    equi_total_end = timer()
    equi_elapsed = (equi_total_end - equi_total_start) / 60
    total_elapsed += equi_elapsed

    print('Time elapsed for equilibrium tests: ', equi_elapsed, 'Minutes')

if sensitivity:
    #setting up the configs for the sensitivity test
    sensitivity_percentage, equi_mode, sens_model_count, sens_time_lst, sens_error_file, sens_error_cnt = \
        sens.config(sensitivity_percentage=0.1,equi_mode=False)
    # launch the init settings for the sensitivity test
    sens.init(folder,sensitivity_percentage)
    # launch the init for the fileops folder, technically not necessary if it has been done in a previous test,
    # but since it's just setting a couple of params, it is not that relevant
    fops.init(folder)

    sens_total_start = timer()

    # File Management
    mdl_files = fops.load_files('py')
    # possibly the file management should go outside of the test, iterating tests over one model
    # instead of iterating models over one test
    # might prove easier for multiple processing
    for model in mdl_files[first_file:last_file]:
        print(model)
        model_name = model.split('.')[0]
        # individual models are timed as well, result is saved in the time_df for this test
        sens_ind_start = timer()
        sens_model_count += 1
        try:
            err_lst = sens.sensitivity(model,equi_mode)
        except Exception as e:
            fops.write_error_file(model_name,e,sens_error_file)
            # if an error happens in the file execution, then the runtime error list is empty and needs to be
            # created and passed as empty
            err_lst = []
            # this goes into the runtime error file to make sure we know that the test script has an error
            err_lst.append(('Incomplete','Incomplete','Incomplete','Incomplete','Incomplete','Incomplete'))
            fops.move_mdl_debug(model,'run_debug')
        #print('Models: ', sens_model_count)
        sens_ind_end = timer()
        # this saves the runtime errors to file
        fops.write_run_error_file(err_lst,'error_file.csv')
        sens_time_lst = fops.read_time_data(model_name, sens_ind_end - sens_ind_start, sens_time_lst)
    fops.save_lst_csv(sens_time_lst,'sens_time','source',columns=['Model Name', 'Time Sens',
                                                                    'Time Horizon', 'Time Step', 'Number of Vars'])

    # calculating the elapsed time for the test and adding it to the total
    sens_total_end = timer()
    sens_elapsed = (sens_total_end - sens_total_start) / 60
    total_elapsed += sens_elapsed

    print('Time elapsed for sensitivity tests: ', sens_elapsed, 'Minutes')

if total_elapsed > 0:
    # this prints the total time, if total_elapsed = 0, no test has been selected
    print('Total time elapsed for entire test:', total_elapsed, 'Minutes')
else:
    print('No test has been selected')