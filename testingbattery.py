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

Version 0.3
Update 30.07.18/sk
"""

import os
import platform
from configparser import ConfigParser
from tb.battery import Battery

# reading config'
config_path = os.path.join(os.getcwd(), '_config', 'tb_config.ini')
cf = ConfigParser()
cf.read(config_path)

# testing settings
# are put in front for convenience
# folder needs to be read in from the config file
test_folder = cf['general config'].get('test_folder',fallback='test')
folder = os.path.join(os.getcwd(), test_folder)
# adjust here for testing
# None means that all files are executed
first_file = None
last_file = None

# testing settings
# testing settings regulate the output
testing_mode = cf['testing'].getboolean('testing_mode', fallback=False)

# pysd helper is a component run before the tests
pysd_helper = cf['component control'].getboolean('PySD_helper', fallback=True)

settings_path = os.path.join(os.getcwd(), '_config', 'settings.ini')
cf.read(settings_path)

# checking platform
if platform.system() == 'Windows':
    # MP currently only works on Windows since it uses a batch file to launch the different tests
    mp_setting = cf['general'].getboolean('mp_setting', fallback=True)
else:
    # to make MP available to other platforms the cmd file needs to be converted
    mp_setting = False

# distance is currently only available in testing mode since it doesn't provide any value just yet
if testing_mode:
    distance = cf['general'].getboolean('set_ko', fallback=True)
    knockout = cf['general'].getboolean('set_dist', fallback=True)
else:
    distance = False
    knockout = False

# if testing mode is active, then first file and last file can be adjusted, if non-testing, it's always all models
if not testing_mode:
    # DO NOT CHANGE
    # this overwrites the first file, last file setting from testing if testing mode is not on
    first_file = None
    last_file = None

# launches battery
bat = Battery(folder, mp_setting=mp_setting, first=first_file, last=last_file, distance=distance, knockout=knockout)
test_cnt = bat.initialize_battery()
# we only launch the run pipe if there are tests to run
if test_cnt != 0:
    bat.create_exec_files()
    bat.run_pipe()
else:
    print('no tests to run')
