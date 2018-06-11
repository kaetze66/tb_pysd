"""
Master file for the testing battery

defines the testing folder (standard: .\test
reads settings from the config file
- test mode
- pysd helper settings

determines the MP setting (only available in windows)
limits the files tested (in testing mode)

initializes battery
runs pipe according to MP setting

Version 0.2
Update 11.06.18/sk
"""

import os
import platform
from configparser import ConfigParser
from tb.battery import Battery

# testing settings
# are put in front for convenience
folder = os.path.join(os.getcwd(), 'test_mode')
# adjust here for testing
first_file = None
last_file = None

# reading config
config_path = os.path.join(os.path.split(folder)[0], '_config', 'tb_config.ini')
cf = ConfigParser()
cf.read(config_path)

# testing settings
# testing settings regulate the output
testing_mode = cf['testing'].getboolean('testing_mode')

# pysd helper is a component run before the tests
pysd_helper = cf['component control'].getboolean('PySD_helper', fallback=True)
# pysd helper only runs pysd helper for .mdl adjustments
pysd_helper_only = cf['component control'].getboolean('PySD_helper_only', fallback=False)

# checking platform
if platform.system() == 'Windows':
    # MP currently only works on Windows since it uses a batch file to launch the different tests
    MP_setting = True
else:
    # to make MP available to other platforms the cmd file needs to be converted
    MP_setting = False

# distance is currently only available in testing mode since it doesn't provide any value just yet
if testing_mode:
    distance = True
else:
    distance = False

# if testing mode is active, then first file and last file can be adjusted, if non-testing, it's always all models
if not testing_mode:
    # DO NOT CHANGE
    # this overwrites the first file, last file setting from testing if testing mode is not on
    first_file = None
    last_file = None

# launches battery
bat = Battery(folder, MP_setting=MP_setting, first=first_file, last=last_file, distance=distance)
if pysd_helper_only:
    bat.run_pysdhelper()
    # PySD helper creates a pickle file that is used for the report. If pysd helper is run stand-alone,
    # this is not needed and needs to be cleaned up
    report_folder = os.path.join(os.path.split(folder)[0], 'report', os.path.split(folder)[1])
    bat.clean_files(report_folder, '.pickle')
else:
    test_cnt = bat.initialize_battery()
    # we only launch the run pipe if there are tests to run
    if test_cnt != 0:
        bat.create_exec_files()
        bat.run_pipe()
    else:
        print('no tests to run')
