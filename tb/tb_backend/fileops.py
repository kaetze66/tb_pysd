"""
This file includes all required file operations in the testing battery

- csv saving and reading
- plot saves
- etc

# DEPRECATED

only still used in descriptives

Version 0.2
Update 30.07.18/sk
"""

import os
import pandas as pd
import shutil
import pickle


def init(working_folder):
    """
    Defines the globals for fileops
    The only global fileops needs is the folder_dict with all non model specific folders
    folder structure is:
    - source: contains the .mdl, .py files as well as the time files and script error reports (either /test or /models)
    - _report: contains all general reporting for the entire model sample
        - _debug:
            _translation: contains a copy of all models that caused an error in translation
            _runtime: contains a copy of all models that caused an error when run
        - _flag: contains a copy of all models that are flagged in translation
    creates the folder dict as global variable to facilitate the saving of the various information,
    seems to work pretty stable
    :param working_folder:
    :return: none
    """
    global folder_dict

    spl_folder = os.path.split(working_folder)
    folder_dict = {'source': working_folder,
                   'report': os.path.join(spl_folder[0], 'report', spl_folder[1]),
                   'flag': os.path.join(spl_folder[0], 'report', spl_folder[1], '_flag'),
                   'trans_debug': os.path.join(spl_folder[0], 'report', spl_folder[1], '_debug', '_translation'),
                   'run_debug': os.path.join(spl_folder[0], 'report', spl_folder[1], '_debug', '_runtime'),
                   'run_pipe': os.path.join(spl_folder[0], 'run_pipe')}


def output_folder(model_name, active):
    """
    creates the model specific folder structure
    folder structure is:

    - source:
        - base: contains the error file for the model
            - doc: contains the documentation from descriptives.py
            - equi: contains the output from equilibrium.py
            - sens: contains the output from sensitivity.py
            - dist: contains the output from the distance.py
            - ... further tests and results

    :param model_name: filename without file extension
    :param active: active folder, possible inputs are 'sens', 'equi', 'doc'
    :return: none
    """
    # these folders are continuously overwritten for every new model
    folder_dict['base'] = os.path.join(folder_dict['source'], model_name)
    # new tests need to be added here
    folder_types = ['sens', 'equi', 'doc', 'dist', 'swit', 'tstep', 'mc']
    for folder in folder_types:
        output_folder = os.path.join(folder_dict['source'], model_name, folder)
        folder_dict[folder] = output_folder
    # create the active folder if doesn't exist yet
    try:
        os.makedirs(folder_dict[active])
    except:
        pass
    # deletes all files in the active folder to avoid confusion
    olfiles = [f for f in os.listdir(folder_dict[active])]
    for file in olfiles:
        os.remove(os.path.join(folder_dict[active], file))


def initiate_report_folder():
    """
    creates the folder structure for the report folder, this is only run once for the entire session,
    uses information from the folder dict created previously (makes fops.init a requirement for this function)

    also initiates the run_pipe folder for the execution dump
    :return: none
    """
    # we clear the translation, debug and flag folders first
    try:
        shutil.rmtree(folder_dict['report'])
        shutil.rmtree(folder_dict['run_pipe'])
    except:
        pass
    # then we create the report folders again
    os.makedirs(folder_dict['report'])
    os.makedirs(folder_dict['flag'])
    os.makedirs(folder_dict['trans_debug'])
    os.makedirs(folder_dict['run_debug'])
    try:
        os.makedirs(folder_dict['run_pipe'])
    except FileExistsError:
        pass


def clear_files_type(folder, type):
    """
    deletes all files from a folder by type
    use with care particularly when it comes to .py files
    :param folder: string, code for target folder (see .init and .output_folder)
    :param type: string, filetype extension
    :return: none
    """
    # this just makes sure that there is a point to the extension, just to be sure,
    # who knows what might happen if there is no point to anything
    if not type.startswith('.'):
        type = '.%s' % type
    olfiles = [f for f in os.listdir(folder_dict[folder]) if f.endswith(type)]
    for file in olfiles:
        os.remove(os.path.join(folder_dict[folder], file))


def move_mdl_debug(model, type):
    """
    this moves the models to debug for further evaluation
    source is always source folder, as models are always in the source folder
    :param model: model name WITH extension, given that it can be .mdl (to trans_debug and flag) and .py (to run_debug)
    :param type: string, can be 'flag', 'trans_debug' or 'run_debug'
    :return: none
    """
    if type == 'trans_debug':
        shutil.move(os.path.join(folder_dict['source'], model), folder_dict[type])
    else:
        shutil.copy2(os.path.join(folder_dict['source'], model), folder_dict[type])


# Load files creates a list with all models of the type
# used in equilibrium.py and sensitivity.py
def load_files(type):
    """
    Returns a list of files
    type needs to be string
    :param type: string, extension of the files to be loaded
    :return: list of files
    """

    files = [f for f in os.listdir(folder_dict['source']) if f.endswith(type)]
    return files


def save_csv(df, name, test):
    """
    Saves data to .csv, is used for output files that will be rewritten for every iteration
    :param df: Dataframe with the data
    :param name: string, name of the output file
    :param test: originating test (e.g. 'sens', 'equi')
    :return: saved .csv
    """
    # the replace elements are just to make sure that there are no naming error issues
    name = name.replace('.py', '').replace('"', '').replace('/', '').replace('*', '')
    return df.to_csv(os.path.join(folder_dict[test], '%s.csv' % name), index=True, header=True)


def append_csv(df, name, test):
    """
    Same as save_csv(), but appends data to .csv files
    This is used mainly for tracking files, such as time files
    :param df: Dataframe with the data
    :param name: string, name of the output file
    :param test: originating test (e.g. 'sens', 'equi')
    :return: appended .csv
    """
    # the replace elements are just to make sure that there are no naming error issues
    name = name.replace('.py', '').replace('"', '').replace('/', '').replace('*', '')
    # this does the same as save, but adds to csv, can be used for time and error tracking
    if os.path.isfile(os.path.join(folder_dict[test], '%s.csv' % name)):
        with open(os.path.join(folder_dict[test], '%s.csv' % name), 'a') as f:
            df.to_csv(f, header=False)
    else:
        save_csv(df, name, test)


def init_run_error_file(err_lst, name):
    """
    this is a copy of write run error file adjusted to descriptives
    descriptives removes the old error file and starts a new one

    :param err_lst: list with errors
    :param name: name of file
    """

    err_df = pd.DataFrame(err_lst)
    if os.path.isfile(os.path.join(folder_dict['base'], name)):
        os.remove(os.path.join(folder_dict['base'], name))
    err_df.columns = ['Source', 'Error Type', 'Description', 'Location', 'Parameters', 'Change to Base']
    save_csv(err_df, 'error_file', 'base')


def write_error_file(model_name, e, name):
    """
    this is to use scripting error files that originate from code
    :param model_name: name of the model file currently being tested, without file extension
    :param e: incurred exception when executing code
    :param name: name of the error file that the error needs to be reported in
    :return: reported error
    """
    f = open(os.path.join(folder_dict['source'], name), 'a', errors='replace')
    f.write('%s : %s\n' % (str(model_name), str(e)))
    f.close()


def save_lst_csv(list, name, test, columns=None, append=True):
    """
    This does the same as save_csv or append_csv, but takes a list as input and transforms it to a Dataframe

    :param list: list with the data
    :param name: name of the file to be saved
    :param test: originating test (e.g. 'sens', 'equi')
    :param columns: optional, can add columns to the dataframe if necessary
    :param append: boolean, whether or not it should be appended
    :return: saved csv
    """
    try:
        list_df = pd.DataFrame(list)
        if columns is not None:
            list_df.columns = columns
        if append:
            append_csv(list_df, name, test)
        else:
            save_csv(list_df, name, test)
    except:
        pass


def rep_orig(name, rep_tpl):
    # unpacking the tuple to add elements
    a, b, c, d = rep_tpl
    rep_tpl = (name, a, b, c, d, folder_dict['report'].replace(os.path.split(folder_dict['source'])[0], ''),
               folder_dict['base'].replace(folder_dict['source'], ''))
    # creating the report pickle file
    pickle_out = open(os.path.join(folder_dict['report'], '%s_0orig.pickle' % name), 'wb')
    pickle.dump(rep_tpl, pickle_out)
    pickle_out.close()


def rep_work(name, rep_tpl):
    # unpacking the tuple to add elements
    a, b, c, d = rep_tpl
    rep_tpl = (
        name, a, b, c, d, folder_dict['report'].replace(os.path.split(folder_dict['source'])[0], '').lstrip('\\'),
        folder_dict['base'].replace(folder_dict['source'], '').lstrip('\\'))
    # creating the report pickle file
    pickle_out = open(os.path.join(folder_dict['report'], '%s_2work.pickle' % name), 'wb')
    pickle.dump(rep_tpl, pickle_out)
    pickle_out.close()
