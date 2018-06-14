"""
PySD helper prepares a model file for pysd translation and fixes some things that pysd can't handle well:

- deletes comments to avoid charmap issues
- replaces data sourcing with data tables
- eliminates unchangeable variables
- replaces intermediate sections
- replaces delay fixed with delay n to the 100th order
- replaces RANDOM 0 1 functions with RANDOM UNIFORM functions

usage:

from tb.tb_backend.pysdhelper import PysdHelper
folder = r'C:\path\to\FOLDER'
model = PysdHelper(folder,'MODELNAME.mdl')
model.run_helper()

integration:
is integrated in the battery. Each model is treated with the pysd helper before run through the testing battery.

todo:
probably doesn't work with Macros

output:
Output is a vensim compatible adjusted model.

To create a xlsx file with all the data go to Model --> Export Dataset and export the data.vdf file.
Make sure under the option "Time Running" the check is set at "down".

Version 0.2
Update 14.06.18/sk

"""

import os
import pandas as pd
import numpy as np
import shutil
from operator import itemgetter
import re
import pickle
from configparser import ConfigParser


class PysdHelper:
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename

        self.data_folder = os.path.join(self.folder, '_data')
        self.orig_folder = os.path.join(self.folder, '_original')
        # models should never be named "anything_treated.mdl", otherwise the testing battery won't pick them up.
        self.outname = '%s_treated.mdl' % self.filename.replace(' ', '').replace('-', '').rsplit('.', 1)[0]
        self.datafile = 'input_%s.xlsx' % filename.rsplit('.', 1)[0]
        self.el_lst = []
        # separators should be fairly robust by now
        self.sectionsep = r'\\\---/// Sketch information - do not modify anything except names'
        self.eqsep = '|'
        self.elsep = '~'
        # list of functions with an init statement, are two groups due to different locations
        self.regfunc_lst = ['INTEG', 'DELAY1I', 'DELAY3I', 'SMOOTHI', 'SMOOTH3I']
        self.nfunc_lst = ['DELAY N', 'SMOOTH N']
        # missing data needs to be shown at the end
        self.missingdata = []
        # preparing the section list for deconstructing and reconstructing file
        self.section_lst = []
        # the data file is also initialized here
        # this could be done better 14.06.18/sk
        self.data = None
        # reading in the config file data
        # config file is defined and read in here
        self.cf = ConfigParser()
        # config folder doesn't change, so it's put here to it's on one line
        self.cf.read(os.path.join(os.path.split(folder)[0], '_config', 'tb_config.ini'))
        # replace init replaces all initial settings for stocks, delay and smooth functions with variables
        # to make it easier to handle and adjust
        self.repl_init = self.cf['PySD helper control'].getboolean('replace_init', fallback=True)
        # assumes that datasource exists, if not, warning is given
        self.data_src = self.cf['PySD helper control'].getboolean('replace_data', fallback=True)
        self.set_rand = self.cf['PySD helper control'].getboolean('replace_random', fallback=True)
        self.set_saveper = self.cf['PySD helper control'].getboolean('replace_saveper', fallback=True)
        self.del_comments = self.cf['PySD helper control'].getboolean('delete_comments', fallback=True)
        self.set_DF = self.cf['PySD helper control'].getboolean('replace_DF', fallback=True)

    # helper methods #

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

    # iteration methods #

    @staticmethod
    def pop_comments(el):
        """
        Sometimes comments produce charset errors if there's something funky in one comment.
        Since we don't really need them, we take them out.

        Input is:
        [varname=expr,unit,comment,[supplementary]]

        Output is:
        [varname=expr,unit]

        :param el: list of element with variable information
        :return: list of element with removed comment and supplementary designation
        """

        if len(el) == 4:
            # if the element length is 4, there's also a supplementary indication, so we also remove that
            el.pop(-1)
            el.pop(-1)
        elif len(el) == 3:
            el.pop(-1)
        else:
            pass
        return el

    @staticmethod
    def elim_unchangeable(el):
        """
        Vensim has the option to make a variable unchangeable which is represented by the '==' syntax.
        Since pysd cannot handle that and it's a bit senseless, it is removed."

        Input is:
        [varname==expr,unit]

        Output is:
        [varname=expr,unit]

        :param el: list of element with variable information
        :return: list of element with removed double equal sign
        """
        el[0] = el[0].replace('==', '=')
        return el

    def data_input_conversion(self, el):
        """
        PySD cannot handle the data inputs from Vensim so this function replaces the data inputs
        with table functions of the same data.

        The script searches for the key word :INTERPOLATE: to trigger data conversion.
        Other data input methods are currently not supported.

        Input is:
        [varname:INTERPOLATE:,unit]

        Output is:
        [varname=DATATABLE varname(Time),unit],
        [DATATABLE varname(Data),unit]

        :param el: list of element with variable information
        :return: list with table function replacing the :INTERPOLATE:, new table variable
        """
        try:
            el[0] = el[0].replace(':INTERPOLATE:', '')
            name = el[0].strip()
            x = self.data.index
            # if the data is properly exported from Vensim, there should not be a key problem
            y = self.data[el[0].strip()]
            # combining the two data series
            table_lst = list(zip(x, y))
            # removing the pairs where there is no y value (pair[1])
            table_lst = [pair for pair in table_lst if not np.isnan(pair[1])]
            # defining the table frame
            table_frm = [(min(table_lst, key=itemgetter(0))[0], min(y)), (max(table_lst, key=itemgetter(0))[0], max(y))]
            # setting the unit of the table variable to the same unit as the source variable
            unit = el[1]
            # setting the name for the source variable
            el[0] = '\n\n%s=\n\tDATATABLE %s(Time)\n\t' % (name, name)
            table_str = ''
            for coord in table_lst:
                # adding a newline and tab to each table coordinate to ensure that the string doesn't get
                # too long because Vensim has trouble with that.
                ncoord = ',\n\t%s' % str(coord).replace(' ', '')
                table_str += ncoord
            table_expr = '\n\nDATATABLE %s(\n\t[%s-%s]%s)\n\t' % (name, table_frm[0], table_frm[1], table_str)
            newel = [table_expr, unit]
            return el, newel
        except KeyError:
            # if the variable is not found in the excel, then the variables are gathered and returned.
            # will require to be run again
            self.missingdata.append(el[0].replace(':INTERPOLATE:', '').strip())
            print(self.missingdata)
            return el, None

    @staticmethod
    def repl_df(el):
        """
        PySD cannot handle delay fixed at the moment so they are replaced with delay n to the 100th order.
        This might make graph interpretations a bit more difficult but it's the best we can do at the moment.
        Delay fixed are unfortunately quite popular.

        Input is:
        [varname=DELAY FIXED(Input,DelayTime,Init),Unit]

        Output is:
        [varname=Delay N(Input,DelayTime,Init,100),Unit]

        :param el: list of element with variable information
        :return: list of element with corrected Delay Fixed
        """
        var, in_expr = el[0].split('=')
        in_expr = in_expr.split('(', 1)[1]
        in_expr = in_expr.rsplit(')', 1)[0]
        # the re.split only splits top level commas here. This ensures that if the input or
        # input element is a function, they are not touched.
        exp_lst = re.split(',\s*(?![^()]*\))', in_expr)
        # order is a bit random but I haven't come across any model where a delay order of 100 has been used.
        # this could be handled as discussed here. https://github.com/JamesPHoughton/pysd/issues/147
        order = 100
        exp_lst.append(str(order))
        func_str = ', '.join(exp_lst)
        el[0] = '%s=\n\tDELAY N(%s)\n\t' % (var, func_str)
        return el

    @staticmethod
    def fix_rand(el):
        """
        In some standard testing structures, RANDOM 0 1 is used. PySD cannot read this but Random Uniform
        performs the same.

        Input is:
        [varname=RANDOM 0 1 (),Unit]

        Output is:
        [varname=RANDOM UNIFORM (0,1,0)

        :param el: list of element with variable information
        :return: fixed list of elements
        """
        var, in_expr = el[0].split('=')
        in_expr = in_expr.replace('RANDOM 0 1', 'RANDOM UNIFORM')
        # this is sketchy at best
        # the reason why the () replacement is separate is due to fact that there might be a
        # new line character in between the function and the ()
        in_expr = in_expr.replace('()', '(0, 1, 0)')
        el[0] = '%s=\n\t%s' % (var, in_expr)
        return el

    @staticmethod
    def fix_saveper(el):
        """
        This change is testing battery specific. If saveper is not equal to timestep some tests won't perform equally.

        Setting will be included in config file.

        :param el: SAVEPER element
        :return: fixed SAVEPER element
        """
        el[0] = '\n\nSAVEPER  = \n\tTIME STEP\n\t'
        return el

    def replace_init(self, el, init_cnt):
        """
        This makes the committed replace init function in vensim2py obsolete (if pysdhelper is run).
        Also has some small improvements that are not used in the vensim2py file
        (e.g. functions in init statements are moved to new variable as well.)

        This is run indiscriminately over all variables, unlike the other adjustments because checking for all
        different elements is a bit inconvenient.

        Input is:
        [varname=INTEG(expr,0),Unit]

        Output is:
        [varname=INTEG(expr,init varname),Unit],
        [init varname = 0,Unit]

        OR

        Input is:
        [varname=INTEG(expr,MAX(x,y)),Unit]

        Output is:
        [varname=INTEG(expr,init varname),Unit],
        [init varname = MAX(x,y),Unit]

        :param init_cnt: int, count of inits replaced for reporting
        :param el: list of element with a init expression
        :return: fixed list of element, new init element
        """
        try:
            # try block avoids errors when table functions are tested (they don't have a '=')
            var, expr = el[0].split('=', 1)
            # if it's the first element it has the encoding information and this can be removed
            # Vensim adds it again if it's missing
            var = var.replace('{UTF-8}', '')
        except ValueError:
            expr = ''
            var = el[0]
        try:
            # try block avoids errors when handling for example constants (they don't have a '(')
            func, expr = expr.split('(', 1)
        except ValueError:
            func = ''
        if func.strip() in self.regfunc_lst or func.strip() in self.nfunc_lst:

            # removing the closing bracket
            expr = expr.rsplit(')', 1)[0]
            # The code below splits first level commas, but not commas within brackets, such as in MAX ( X , Y )
            # and returns 3 or 4 elements depending on which function
            # This is separated in order to be able to replace the init element and put it back together
            expr_split = re.split(',\s*(?![^()]*\))', expr)
            if func.strip() in self.regfunc_lst:
                pos = -1
            elif func.strip() in self.nfunc_lst:
                pos = -2
            else:
                pos = None
            init = expr_split[pos]
            # if the init expression has a comma, then it's a function and should be stored in a
            # separate variable as well
            if self.constant(init) or ',' in init:
                init_cnt += 1
                # init expression gets a newline character to ensure that string doesn't get too long
                init_expr = '\n\tinit %s' % var.strip()
                expr_split[pos] = init_expr
                expr = ', '.join(expr_split)
                el[0] = '%s=%s(%s)\n\t' % (var, func, expr)
                # creating the init element with the same unit and the value or function stored
                init_el = ['\n%s=\n\t%s\n\t' % (init_expr.strip(), init), el[1]]
                # they are appended at the end of the list, so come in after the builtin variables, but it works
                self.el_lst.append(init_el)
        return el, init_cnt

    # pysd helper methods #

    def get_elements_list(self):
        """
        Reads in the .mdl file and separates the elements. First split by section, then equation, then elements.

        Input is:
        '{UTF-8}\nvarname=\n\texpr\n\t~\t unit\n\t~\n\tcomment~\n\tsupplementary|...

        Output is:

        [varname=expr,unit,comment,supplementary]

        :return: list of elements
        """
        with open(os.path.join(self.folder, self.filename), 'r', encoding='utf8', errors='replace') as in_file:
            text = in_file.read()
        # splitting the sections off
        self.section_lst = text.split(self.sectionsep)
        # splitting the equations
        eq_lst = self.section_lst[0].split(self.eqsep)
        # last equation element is empty
        eq_lst.pop(-1)
        for el in eq_lst:
            # splitting the elements
            spel = el.split(self.elsep)
            self.el_lst.append(spel)
        # removing the section separators (for views, etc.)
        self.el_lst = [el for el in self.el_lst if
                       '********************************************************' not in el[0]]

    def read_data(self):
        """
        Tries to read data in and returns a DF if source exists.

        If source doesn't exist, it provides a message with indications. Message shows up even if no data is needed.

        :return: DF with data or warning
        """
        try:
            self.data = pd.read_excel(os.path.join(self.data_folder, self.datafile), index_col=0)
        except FileNotFoundError:
            print('''No data file found. Please put a file with the name '%s' in the _data folder. ''' % self.datafile)
            # create folder if it doesn't exist
            if not os.path.isdir(self.data_folder):
                os.makedirs(self.data_folder)
            # if no data exists, then we don't do data replacement
            self.data_src = False

    def iterate_elements(self):
        """
        Iterating through all the elements and executing the different changes

        :return:
        """
        # setting up the counts
        unchang_cnt, data_cnt, dfixed_cnt, random_cnt, init_cnt = 0, 0, 0, 0, 0
        # we should also report which variables that the delay fixed has been changed in
        dfixed_lst = []
        for el in self.el_lst:
            # deleting commments and unchangeables are run over all variables
            if self.del_comments:
                el = self.pop_comments(el)
            # this needs to be tested 05.06.18/sk
            if '==' in el[0]:
                unchang_cnt += 1
                el = self.elim_unchangeable(el)
            # data, DF, Random and SavePer are only executed based on keywords
            if self.data_src:

                if ':INTERPOLATE:' in el[0]:
                    data_cnt += 1
                    el, newel = self.data_input_conversion(el)
                    if newel is not None:
                        self.el_lst.append(newel)
            if self.set_DF:
                if 'DELAY FIXED' in el[0]:
                    dfixed_cnt += 1
                    dfixed_lst.append(el[0].split('=')[0])
                    el = self.repl_df(el)
            if self.set_rand:
                # this needs to be tested
                if 'RANDOM 0 1' in el[0].replace('\n', ''):
                    random_cnt += 1
                    el = self.fix_rand(el)
            if self.set_saveper:
                if 'SAVEPER' in el[0]:
                    el = self.fix_saveper(el)
            # replace init is run over all variables again
            el, init_cnt = self.replace_init(el, init_cnt)
            # report tuple is created for the reporting
            rep_tpl = ('PySD Helper', [unchang_cnt, data_cnt, dfixed_cnt, random_cnt, init_cnt], dfixed_lst,
                       os.path.join(os.path.split(self.orig_folder)[1], self.filename).lstrip(
                           '\\'), self.outname)
            folder_lst = os.path.split(self.folder)
            # report tuple is pickled and then used in initiate report in the battery
            pickle_out = open(os.path.join(folder_lst[0], 'report', folder_lst[1],
                                           '%s_1helper.pickle' % self.outname.rsplit('.', 1)[0]), 'wb')
            pickle.dump(rep_tpl, pickle_out)
            pickle_out.close()

    def write_file(self):
        """
        Rewrites the model file with the changes, maintains Vensim syntax to ensure that model can
        still be run in Vensim.

        :return: Treated mdl file
        """
        with open(os.path.join(self.folder, self.outname), 'w') as in_file:
            for el in self.el_lst:
                # adding the pieces back together
                for i in el:
                    in_file.write(i)
                    in_file.write('%s' % self.elsep)
                in_file.write('\n\t%s' % self.eqsep)
            in_file.write('\n\n%s' % self.sectionsep)
            # adding the Vensim stuff at the end
            in_file.write(self.section_lst[1])

    def move_orig(self):
        """
        saves the original model file in the _original folder just to be sure
        :return:
        """
        if not os.path.isdir(self.orig_folder):
            os.makedirs(self.orig_folder)
        shutil.move(os.path.join(self.folder, self.filename), os.path.join(self.orig_folder, self.filename))

    def run_helper(self):
        """
        Running the methods in the right order

        :return:
        """
        self.get_elements_list()
        self.read_data()
        self.iterate_elements()
        self.write_file()
        self.move_orig()
