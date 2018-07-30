"""
This code translates .mdl files and produces
    - csv file: detailed descriptives of all variables
    - doc file: file with information used later in the testing battery
    - equi file: creates file for user input for equilibrium test
    - py file: translated .mdl file using pysd
    - model stats: model statistics of all files translated
    - word analysis files: experimental code for later use
    - other testing files: collecting functions, gathering errors, etc.

The descriptives add additional detail compared to pysd and
permit a quick review of the variables in an excel file

Some sections are closely mirrored from pysd

This needs to be drastically improved 30.07.18/sk

version 0.2
30.07.18/sk
"""

import os
import pysd
import pandas as pd
import re
from timeit import default_timer as timer
from collections import Counter
from tb.tb_backend import utils
from tb.tb_backend import fileops as fops


def flow_split(expr):
    """
    splitting the stock expressions to extract the flow namespace

    expr is INTEG (inflow-outflow,init stock)

    returns:
    - flows: ['inflow', 'outflow']
    - flow: inflow-outflow
    - init: init stock)

    :param expr: expr expression from translation routine (everything right of the equation sign)
    :return: list of flows, flow expression and init expression
    """
    # eliminating the INTEG ( part
    expr = expr.split('(', 1)[-1]
    # splitting the init off the expression
    flow, init = expr.rsplit(',', 1)
    # splitting the flows off, but avoiding splits between parentheses
    flows = re.split('(".*?")|[+-/*]', flow)
    # removing empty strings in flows
    flows = [x for x in flows if x is not None]
    flows = [x for x in flows if x != '']
    flows = [x.strip() for x in flows]
    # removing the closing parenthesis from the init expression
    init = init.replace(')', '')
    return flows, flow, init


def get_sections(textstring):
    """
    splitting the text string and returning the equation string (uncleaned) for both builtin and elements
    the text string is split in three parts (elements, builtins, garbage)
    garbage is split off based on the string 'Sketch information'
    builtins are split off by string '.Control'
    however, sometimes elements are listed below the .Control split, thus this needs to be cleaned afterwards

    input is:
    {UTF-8}
    exo BL1=
        3
        ~	Month
        ~		|
    exo BL2=
        0.5
        ~	1/Month
        ~		|

    output is:
    ['exo BL1=3~Month~', 'exo BL2=0.5~1/Month~',

    :param textstring: text version of a Vensim model
    :return: list of equation strings (unclean)
    """
    # removing charset info
    basetext = textstring.replace('{UTF-8}', '')
    # replace new line and tab characters
    wtext = basetext.replace('\n', '').replace('\t', '')
    # replace backslashes
    text = wtext.replace('\\', '')
    # split off garbage
    text = text.split('Sketch information')[0]
    # split elements and builtin sections
    sections = text.split('.Control')
    # split elements section into each element
    elements = sections[0].split('|')
    # remove the last element as it is empty
    elements = elements[:-1]
    # the same is done with builtins, but this is done in a try block because for some reason the
    # .Control splitter could not exist (this case actually is in the sample)
    try:
        built_ins = sections[1].split('|')
        built_ins = built_ins[1:-1]
    except:
        # if there is no section, the builtins are empty
        built_ins = []
    return elements, built_ins


def get_vars(varlist):
    """
    splitting the equations and creating the varlist info
    input is:
    ['exo BL1=3~Month~', 'exo BL2=0.5~1/Month~',
    output is:
    [{'eqn': 'exo BL1=3', 'unit': 'Month', 'comment': '', 'name': 'exo BL1', 'expr': '3'},

    is run for both the elements and the builtins

    the inner try except block ensures that table functions are also read in

    :param varlist: list of equation strings (unclean)
    :return: list of dicts with elements and builtins
    """
    components = []
    for element in varlist:
        try:
            eqn, unit, comment = element.split('~', 2)
            eqn = eqn.strip()
            unit = unit.strip()
            comment = comment.strip()
            # name is left of =, expr on the right
            try:
                name, expr = eqn.split('=', 1)
            except ValueError:
                name = eqn
                expr = eqn
            components.append({'eqn': eqn,
                               'unit': unit,
                               'comment': comment,
                               'name': name.strip(),
                               'expr': expr.strip()})
        except ValueError:
            pass
    return components


def corr_lists(varlist, builtins):
    """
    this corrects list in case the split with .Control didn't split correctly
    this could be much easier if we hadn't split for sections before, but this has to be revisited at some other point,
    right now I'm just happy this works

    :param varlist: list of dict with variables
    :param builtins: list of dict with builtins
    :return: clean list of dict with variables and builtins (of which there are 4)
    """
    # is used to ensure that there is no infinite loop, two iterations are enough
    # it is for example possible that (BI = builtin):
    # builtins = [var1] and varlist = [var2, var3, BI1, BI2, BI3, BI4]
    # in that case the function will first move the builtins to the builtin list with the result that
    # builtins = [var1, BI1, BI2, BI3, BI4] and varlist = [var2, var3]
    # now in the second iteration the elif condition applies and the var1 is moved to the varlist, resulting
    # builtins = [BI1, BI2, BI3, BI4] and varlist = [var1, var2, var3]

    i = 0
    while len(builtins) != 4 and i <= 2:
        if len(builtins) < 4:
            translist = [x for x in varlist if x['name'] in ['FINAL TIME', 'INITIAL TIME', 'TIME STEP', 'SAVEPER']]
            varlist = [x for x in varlist if x['name'] not in ['FINAL TIME', 'INITIAL TIME', 'TIME STEP', 'SAVEPER']]
            for item in translist:
                builtins.append(item)
        elif len(builtins) > 4:
            translist = [x for x in builtins if x['name'] not in ['FINAL TIME', 'INITIAL TIME', 'TIME STEP', 'SAVEPER']]
            builtins = [x for x in builtins if x['name'] in ['FINAL TIME', 'INITIAL TIME', 'TIME STEP', 'SAVEPER']]
            for item in translist:
                varlist.append(item)
        i = i + 1
    return varlist, builtins


def id_tables(var, tbl_functions):
    """
    This identifies tables and replaces the table information with the string '(table_expr)' to
    facilitate calculating statistics

    input is:
    {'eqn': 'exo BL1=3', 'unit': 'Month', 'comment': '', 'name': 'exo BL1', 'expr': '3', 'type': 'constant',
        'flow expr': 'NA', 'init expr': 'NA', 'table expr': 'NA'}

    :param var: dict for each variable
    :param tbl_functions: list of tbl_functions
    :return: corrected variable dict
    """

    # function list are put in tuple to be used here
    if var['expr'].startswith(tuple(tbl_functions)):
        # this is the routine if tables are introduced with functions
        var['type'] = 'table function'
        # in this case the table expression is after the last comma
        c = re.split(',', var['expr'], 1)
        c = [x for x in c if x != '']
        var['table expr'] = c[-1]
        var['expr'] = var['expr'].replace(c[-1], '(table_expr))')
    # if tables are introduced without a function it's a bit more complicated
    # the identifiers are that the eqn string does not have an equal sign and
    # that either no : exists (which would be for :AND:) OR the string ([( exists,
    # which is the opening bracket of the table expression
    elif len(var['eqn'].split('=')) == 1 and len(var['eqn'].split(':')) == 1 or len(var['eqn'].split('=')) == 1 \
            and len(var['eqn'].split('([(')) > 1:
        var['type'] = 'table function'
        # in this case the table expression is between two non-greedy parantheses
        c = re.split('(\(.*\))', var['expr'])
        c = [x for x in c if x != '']
        var['table expr'] = c[-1]
        var['expr'] = var['expr'].replace(c[-1], '(table_expr)')
        var['name'] = var['name'].replace(c[-1], '')
    # test with without parentheses 30.07.18/sk
    return var


def get_types(components, tbl_functions, c_functions, s_functions, t_functions):
    """
    defining types for the variables
    input is:
    [{'eqn': 'exo BL1=3', 'unit': 'Month', 'comment': '', 'name': 'exo BL1', 'expr': '3'},...
    [{'eqn': 'exo BL1=3', 'unit': 'Month', 'comment': '', 'name': 'exo BL1', 'expr': '3', 'type': 'constant',
        'flow expr': 'NA', 'init expr': 'NA', 'table expr': 'NA', 'math type': 'regular', 'function': 'NA'},...

    :param components: list of dict with the variables
    :param tbl_functions: list of functions that are used for tables
    :param c_functions: list of complicated functions
    :param s_functions: list of simple functions
    :param t_functions: list of testing functions
    :return:
    """

    flows_list = []
    for entry in components:
        # if the expression is a constant, the type is always constant
        if utils.constant(entry['expr']):
            entry['type'] = 'constant'
            entry['flow expr'] = 'NA'
            entry['init expr'] = 'NA'
            entry['table expr'] = 'NA'
        # if the expression starts with INTEG, it's always a stock
        elif entry['expr'].startswith('INTEG') and not entry['expr'].startswith('INTEGER'):
            entry['type'] = 'stock'
            # flows differ from other auxiliaries that they are the only ones that can impact a stock
            # thus the flow names are saved in a list and changed later
            flows, flow_expr, init_expr = flow_split(entry['expr'])
            for flow in flows:
                if flow not in flows_list and len(flow) > 0:
                    flows_list.append(flow)
            # an init variable list could be created here and init variables could be classified different
            # than constants 06.07.18/sk
            entry['flow expr'] = flow_expr
            entry['init expr'] = init_expr
            entry['table expr'] = 'NA'
        else:
            # everything that is not a constant or a stock, is first typed as an auxiliary
            entry['type'] = 'auxiliary'
            entry['flow expr'] = 'NA'
            entry['init expr'] = 'NA'
            entry['table expr'] = 'NA'
    for entry in components:
        # if the name is in the flow list, it's a flow, the split on [ is just if there are subscripts
        if entry['name'].split('[')[0] in flows_list or entry['name'] in flows_list:
            entry['type'] = 'flow'
        # then tables are identified with the ID tables function
        entry = id_tables(entry, tbl_functions)
        # subscripts need to be named subscripts in this one
    for entry in components:
        # split should be on elements, not on the entire equation
        # this is to define the math types and separate the function types for statistics
        # this only works for functions that are at the beginning of the expression
        try:
            func, expr = entry['expr'].split('(', 1)
        except ValueError:
            func = ''
            expr = ''
        func = func.strip()
        # removing the closing bracket
        expr = expr.rsplit(')', 1)[0]
        # The code below splits first level commas, but not commas within brackets, such as in MAX ( X , Y )
        # and returns 3 or 4 elements depending on which function
        # This is separated in order to be able to replace the init element and put it back together
        expr_split = re.split(',\s*(?![^()]*\))', expr)
        if func == 'A FUNCTION OF':
            entry['math type'] = 'incomplete equation'
            entry['function'] = func
        elif func in c_functions:
            entry['math type'] = 'complicated function'
            entry['function'] = func
            if func in regfunc_lst:
                pos = -1
            elif func in nfunc_lst:
                pos = -2
            else:
                pos = None
            if pos is not None:
                entry['init expr'] = expr_split[pos]
        elif func in s_functions:
            entry['math type'] = 'simple function'
            entry['function'] = func
        elif func in t_functions:
            entry['math type'] = 'testing function'
            entry['function'] = func
        else:
            entry['math type'] = 'regular'
            entry['function'] = 'NA'
    return components


# handling the subscripts

def rem_subscripts(components):
    """
    this removes the subscript brackets and adds them to the sub_expr and subs column in the doc

    this function has to be reviewed and better documented

    :param components: list of dict of the variables
    :return: list of dict of the variables with subscript elements added
    """
    sublist = []
    subdict = {}
    for entry in components:
        if len(entry['eqn'].split('=')) == 1 and len(entry['eqn'].split(':')) > 1:
            subs = re.split(':', entry['expr'])
            entry['expr'] = subs[0]
            entry['sub_expr'] = subs[-1]
            sub_ins = re.split(',', subs[-1])
            ins = len(sub_ins)
            if ins == 1:
                ex = subs[-1].replace('(', '').replace(')', '')
                bounds = re.split('([0-9]*)', ex)
                bounds = [x for x in bounds if utils.constant(x)]
                try:
                    ins = int(bounds[-1]) - int(bounds[0]) + 1
                except:
                    ins = 1
            entry['no of sub_ins'] = ins
            entry['no of subs'] = 1
            entry['type'] = 'subscript list'
            if subs[0] not in sublist:
                sublist.append(subs[0])
                subdict[subs[0]] = len(sub_ins)
        else:
            entry['subs'] = 'NA'
            entry['no of subs'] = 'NA'
            entry['no of sub_ins'] = 1
            entry['sub_expr'] = 'NA'
    for entry in components:
        if len(re.split('(\[.*?\])', entry['name'])) > 1:
            subexpr = re.split('(\[.*?\])', entry['name'])[1]
            subexpr = subexpr.replace('[', '').replace(']', '')
            subs = subexpr.split(',')
            ins = 1
            if set(subs).issubset(sublist):
                entry['subs'] = subs
                entry['no of subs'] = len(subs)
                for sub in subs:
                    ins = ins * subdict.get(sub)
                entry['no of sub_ins'] = ins
            temp_expr = entry['expr'].replace(';', ',')
            temp_els = re.split(',', temp_expr)
            temp_els = [x for x in temp_els if x != '']
            i = 0
            for el in temp_els:
                if utils.constant(el):
                    i = i + 1
            if len(temp_els) == ins and ins == i:
                entry['type'] = 'subscripted constant'

    for entry in components:
        if entry['no of subs'] != 'NA':
            # should probably remove emtpies here
            if entry['no of subs'] == 1 and entry['math type'] == 'regular' and len(entry['expr'].split(',')) > 1:
                entry['det_sub_ins'] = len(entry['expr'].split(','))
            elif entry['no of subs'] > 1 and entry['math type'] == 'regular' and len(entry['expr'].split(';')) > 1:
                entry['det_sub_ins'] = len(entry['expr'].split(';'))
            else:
                entry['det_sub_ins'] = 'NA'
        else:
            entry['det_sub_ins'] = 'NA'
    return components


def collect_functions(components, collection, missing):
    """
    collects additional functions that haven't been sorted
    also collects garbage, but it's good enough

    this could potentially be eliminated as the collection happens better in equation split

    :param components: list of dict with variables
    :param collection: list of functions that have been sorted
    :param missing: list of functions that haven't been sorted from previous iterations
    :return: list of functions updated with new functions
    """
    for entry in components:
        func = entry['expr'].split('(')[0].strip()
        func = re.split("[+-/*]", func)[-1].strip()
        if func.isupper():
            if len(func) < len(entry['expr'].strip()) and func not in collection and func not in missing:
                missing.append(func)
    return missing


def equation_split(varlist, funclist, missing, t_function):
    """
    this splits the equation into its elements and counts them and saves some information based on types to the varlist

    :param varlist: list of dict of the variables
    :param funclist: list, combined with all functions
    :param missing: list, missing functions
    :param t_function: list, testing functions
    :return: list of dict with updated information
    """
    for var in varlist:
        e = re.split('(".*?")|\+|-|\*|/|\(|\)|\^|,|>|<', var['expr'])
        e = [x for x in e if x is not None]
        e = [x.strip() for x in e]
        e = [x for x in e if x != '']
        # m collects the missing functions (statements in upper case) that are not at the beginning of the expression
        m = [x for x in e if x.isupper()]
        # f collects functions even if they are not upper case
        f = [x for x in e if x.upper() in funclist]
        e = [x for x in e if x.upper() not in funclist]
        for func in m:
            # the functions that are not already in the list or already collected in missing are added
            if func not in funclist and func not in missing:
                missing.append(func)
        # types and subscripted constants don't need further information
        if var['type'] == 'constant' or var['type'] == 'subscripted constant':
            nbr, e, hasinit = 0, [], 'NA'
        # stocks have additional information here (number of elements and hasinit)
        elif var['type'] == 'stock':
            e = [x for x in e if x != 'INTEG']
            nbr = len(e) - 1
            if utils.constant(var['init expr']):
                hasinit = 'no'
            else:
                hasinit = 'yes'
        else:
            nbr = len(e)
            hasinit = 'NA'
        var['Number of elements'] = nbr
        var['INIT'] = hasinit
        var['elements'] = e
        var['function list'] = f
        funcs = len(f)
        if funcs > 0 and var['math type'] == 'regular':
            var['function'] = f[0]
            if f[0] in t_function:
                var['math type'] = 'testing function'
            else:
                var['math type'] = 'simple function'
        var['no of functions'] = funcs
        # splitting the elements in the init expression because we need them for the loop recognition
        if var['init expr'] != 'NA':
            ie = re.split('(".*?")|\+|-|\*|/|\(|\)|\^|,|>|<', var['init expr'])
            ie = [x for x in ie if x is not None]
            ie = [x.strip() for x in ie]
            ie = [x for x in ie if x != '']
            var['init elements'] = ie
        else:
            var['init elements'] = []
    return varlist


def add_builtin(varlist, builtin, model_name):
    """
    this adds the builtin information to the variables for statistical purposes

    :param varlist: list of dicts with variables
    :param builtin: list with builtins
    :param model_name: str with model name
    :return: list of dicts with builtin information added
    """

    # bunit: base unit is interesting when other time units are in the model
    # SAVEPER is irrelevant and thus ignored
    ftime, bunit, itime, tstep = '', '', '', ''
    for item in builtin:
        if item['name'] == 'FINAL TIME':
            ftime = item['expr']
            bunit = item['unit']
        elif item['name'] == 'INITIAL TIME':
            itime = item['expr']
        elif item['name'] == 'TIME STEP':
            tstep = item['expr']
    for var in varlist:
        var['FINAL TIME'] = ftime
        var['Base Unit'] = bunit
        var['INITIAL TIME'] = itime
        var['TIME STEP'] = tstep
        var['Model name'] = model_name
    return varlist


def corr_units(varlist):
    """
    The unit correction is necessary as in the case when there are subscripts,
    the units are not associated correctly with all the subscript instances

    therefore the same unit needs to be passed down to other instances

    unit_dict is not used anywhere else, but could be:
    {'exo BL1': 'Month', 'exo BL2': '1/Month', 'exo RL1': 'Month', ...

    :param varlist: list of dict of the variables
    :return: list of dict of the variables with corrected units
    """
    unit_dict = {}
    for var in varlist:
        if var['unit'] != '':
            unit_dict[var['name'].split('[')[0]] = var['unit']
    for var in varlist:
        if var['type'] != 'subscript list' and var['unit'] == '':
            var['unit'] = unit_dict.get(var['name'].split('[')[0])
    return varlist


def calc_avg(varlist):
    """
    Collecting the statistics for descriptives, including number of elements, number of functions,
    number of variables, number of constants

    :param varlist: list of dict of the variables
    :return: total variables, average elements per equation, number of functions and average, constants, empty units
    """
    tot, els, avg, funcs, cons, e_unit, sl_tot = 0, 0, 0, 0, 0, 0, 0
    # two different count types, once with subscript and once without (i.e. number of elements with
    # and without subscripts)
    for var in varlist:
        if var['type'] != 'constant' and var['type'] != 'subscripted constant':
            tot = tot + 1 * var['no of sub_ins']
            els = els + var['Number of elements'] * var['no of sub_ins']
            funcs = funcs + var['no of functions'] * var['no of sub_ins']
        if var['type'] == 'constant' or var['type'] == 'subscripted constant':
            cons = cons + 1 * var['no of sub_ins']
        if var['type'] != 'subscript list':
            sl_tot = sl_tot + 1
            if var['unit'] is None:
                e_unit = e_unit + 1
    try:
        avg = els / tot
        f_avg = funcs / tot
        unit_per = e_unit / sl_tot
    except ZeroDivisionError:
        avg = 0
        f_avg = 0
        unit_per = 1
    return tot, avg, funcs, f_avg, cons, unit_per


def word_analysis(varlist, worddict, wordlist):
    """
    Function to collect word use in models, collects a stream of words and a dictionary with number of uses

    :param varlist: list of dict of the variables
    :param worddict: dictionary with the word counts
    :param wordlist: used words in the models
    :return: word data
    """
    for var in varlist:
        w_name = re.sub('\[.*?\]', ' ', var['name'])
        w_name = re.sub('\"', ' ', w_name)
        w_name = re.sub(':', ' ', w_name)
        w_name = re.sub('\(table_expr\)', ' ', w_name)
        w_list = w_name.split()
        for w in w_list:
            w = w.strip()
            # all words are capitalized to make sure no double counting happens
            w = w.upper()
            if w in worddict:
                worddict[w] = worddict[w] + 1
            else:
                worddict[w] = 1
        if var['unit'] is not None:
            # btu = base time unit, this is to indicate that when a time unit is used, it's not the base
            # and multiple time units are used in the model
            w_unit = var['unit'].replace(var['Base Unit'], 'btu')
        else:
            w_unit = None
        wordlist.append({'unit': w_unit,
                         'words': w_list})
    return worddict, wordlist


def flagging(type_counter, func_counter, empty_stocks, varlist, builtins):
    """
    This function flags models with obvious errors and codifies the error

    :param type_counter: collections.counter object with the type information of the model
    :param func_counter: collections.counter object with the function information of the model
    :param empty_stocks: int, count of empty stocks from collect stats
    :param varlist: list of dict of the variables
    :param builtins: list of dict of the builtins
    :return: flag and code for the
    """

    flag = 'No'
    code = 'None'

    # if the length of the varlist is 0, the model is empty
    if len(varlist) == 0:
        flag = 'Yes'
        code = 'Empty'
    # if the length of the builtin is not 4, there is a problem with the builtin
    elif len(builtins) != 4:
        flag = 'Yes'
        code = 'Builtin'
    # if the count of functions with 'A FUNCTION OF' is higher than 0, there is a problem with some equations
    elif func_counter['A FUNCTION OF'] > 0:
        flag = 'Yes'
        code = 'incomplete equations'
    # if there are empty stocks (i.e. stocks with a constant in it and nothing else), there is a problem
    elif empty_stocks > 0:
        flag = 'Yes'
        code = 'constant stocks'
    try:
        # if the number of flows is smaller than 0.5, the model is obviously wrong
        # (there needs to be at least 0.5 flows per stock)
        # could be replaced with fs ratio from collect stats, but would require a
        # division by zero check anyway for next flag, so why bother?
        if type_counter['flow'] / type_counter['stock'] < 0.5:
            flag = 'Yes'
            code = 'flow recognition'
    # if the previous test returns a zerodivision error, then there are no stocks in the model
    except ZeroDivisionError:
        flag = 'Yes'
        code = 'No Stocks'
    return flag, code


def doc(doc_name, doc_vars, model_doc):
    """
    this creates the doc folder with all the information that is used in further tests and
    is a selection of the descriptives

    The doc file uses the following columns:

    - Base Unit: for the x axis of plots
    - flow expr: for the equilibrium function
    - type: for the different type df (switches are not a type here)
    - Real Name: For output lists
    - Py Name: For input dicts
    - elements: for distance calculations
    - TIME STEP: for the integration test
    - function: used for init replacement in doc file
    - Unit: used for plots

    :param doc_name: name of the doc file
    :param doc_vars: full descriptive database
    :param model_doc: doc from pysd
    :return: saved doc file
    """

    def fill_blanks(row):
        """
        Function to fill the blanks for the builtin variables coming from model.doc()

        :param row: row to fill with NA where empty
        :return: row: NA filled row
        """
        # the builtins and init variables need to be added back to the list now for the doc,
        # they are coming from the model.doc() from pysd because init variables are only created there
        builtin_list = ['FINAL TIME', 'INITIAL TIME', 'TIME STEP', 'SAVEPER']
        if pd.isnull(row['type']):
            if row['Real Name'] in builtin_list:
                row['type'] = 'builtin'
            elif row['Real Name'].startswith('init'):
                row['type'] = 'constant'
            else:
                # if there is an undef type, something is wrong
                row['type'] = 'undef'
            # here we just fill the remaining columns as they are irrelevant for both the builtins and the
            row['flow expr'] = 'NA'
            row['elements'] = []
            row['init elements'] = []
            row['function list'] = []
            row['expr'] = 'NA'
            row['table expr'] = 'NA'
            row['Base Unit'] = doc_vars.iloc[0]['Base Unit']
        return row

    # these are the dropped columns because they are not used from descriptives
    # last line is used for testing the _doc columns
    drop_cols = ['INIT', 'eqn', 'unit', 'comment', 'init expr', 'math type',
                 'subs', 'no of subs', 'no of sub_ins', 'sub_expr', 'det_sub_ins', 'Number of elements',
                 'no of functions', 'FINAL TIME', 'INITIAL TIME', 'Model name',
                 'TIME STEP', 'function']
    doc_vars.drop(drop_cols, axis=1, inplace=True)
    # merge with model.doc() from pysd to get the py names in the doc
    doc_vars = pd.merge(left=doc_vars, right=model_doc, how='outer', left_on='name', right_on='Real Name')
    # drop columns that are not used from model.doc()
    drop_cols = ['Type', 'Comment', 'name']
    doc_vars.drop(drop_cols, axis=1, inplace=True)
    doc_vars.apply(fill_blanks, axis=1)
    fops.save_csv(doc_vars, doc_name, test)
    return doc_vars


def collect_stats(model_vars):
    """
    Collecting the model stats for current model

    This function creates counters for:
    - type: type of variable (i.e. constant, auxiliary, etc.)
    - math type: math type of variable (i.e. regular, simple function, etc.)
    - INIT: whether or not a stock has an init variable
    - function: which function is used (only first in the equation)

    it also checks for empty stocks, i.e. when the flow expression of a stock is a constant

    then it also collects the number of subscript instances, i.e. if a subscript has 10 instances,
    an auxiliary with that subscript counts for 10 variables

    also checks the flow stock ratio

    :param model_vars: list of dict of the variables
    :return: counters, number of empty stocks, subscript counts for stocks, auxiliaries and flows, flow stock ratio
    """

    # counter for types
    c = Counter()
    # counter for math types
    d = Counter()
    # counter for INIT
    e = Counter()
    # counter for functions
    f = Counter()
    emstocks = 0
    for var in model_vars:
        c[var['type']] += 1
        d[var['math type']] += 1
        e[var['INIT']] += 1
        f[var['function']] += 1
        if utils.constant(var['flow expr']):
            emstocks += 1
    s_stocks = sum(x['no of sub_ins'] for x in model_vars if x['type'] == 'stock')
    s_aux = sum(x['no of sub_ins'] for x in model_vars if x['type'] == 'auxiliary')
    s_flow = sum(x['no of sub_ins'] for x in model_vars if x['type'] == 'flow')
    try:
        fs_ratio = c['flow'] / c['stock']
    except ZeroDivisionError:
        fs_ratio = 0
    return c, d, e, f, emstocks, s_stocks, s_aux, s_flow, fs_ratio


def list_combine():
    """
    combining the lists of functions for general function list checks

    :return: combined list of functions
    """

    for f in tbl_func_list:
        comb_func_list.append(f)
    for f in comp_func_list:
        comb_func_list.append(f)
    for f in simp_func_list:
        comb_func_list.append(f)
    for f in test_func_list:
        comb_func_list.append(f)


def init(folder):
    """
    Creates the globals for descriptives and handles preparatory file operations in the base and report folder

    :param folder: string, path to the base folder coming from the testingbattery
    :return:
    """
    fops.init(folder)
    # defining the target folder
    # current directories are test and models
    global test
    test = 'doc'
    global source_folder
    source_folder = folder
    # clear and create structure of the report folder, happens in fileops
    fops.initiate_report_folder()

    # then we remove the .py files from the source folder
    # could potentially be an append function, but this is going to be run after hopefully extensive model changes,
    # so why bother?
    fops.clear_files_type('source', '.py')

    # then we remove the .csv files, mainly time tracking files
    fops.clear_files_type('source', '.csv')

    # combine the lists
    list_combine()


def descriptives(mdl_file, flag_count, word_dict, word_list, mis_func_list, rep_type='work'):
    """
    This function runs the descriptives for an individual file and is called from the full translate

    outputs are:
    - descriptive file
    - list of dict with variables
    - flag related information
    - word analysis information
    - missing function collection

    :param rep_type:
    :param mdl_file: string, .mdl file to be run through descriptives
    :param flag_count: int, number of flagged models previous to this one
    :param word_dict: dict, word dictionary with words previous to this one
    :param word_list: list, words used previous to this one
    :param mis_func_list: list, missing functions previous to this one
    :return: list of dict with vars, flag count, word files, missing functions list
    """
    model_name = mdl_file.split('.')[0]

    # code specific to the ISDC sample, returns garbage in other cases
    year = re.split('_', model_name)[0]

    ind_start = timer()
    with open(os.path.join(source_folder, mdl_file), 'r', encoding='utf8', errors='replace') as in_file:
        text = in_file.read()

    # extracting the sections from the text string and get the variables in two sets, model vars and builtins
    # builtins may fail or contain also model vars
    var_list, built_ins = get_sections(text)
    model_vars = get_vars(var_list)
    built_ins = get_vars(built_ins)
    # correct the lists, builtins should end up being 4 variables
    model_vars, built_ins = corr_lists(model_vars, built_ins)
    # determining the typology of the variables
    model_vars = get_types(model_vars, tbl_func_list, comp_func_list, simp_func_list, test_func_list)
    # removing subscripts from the expr
    model_vars = rem_subscripts(model_vars)
    # split the equations to get the elements of the equation
    model_vars = equation_split(model_vars, comb_func_list, mis_func_list, test_func_list)
    # collect missing functions to be added to the function list
    # currently collects lots of garbage and misses functions that are not in first place of the eqn string
    mis_func_list = collect_functions(model_vars, comb_func_list, mis_func_list)
    # add builtins to model vars to save space (builtins only relevant with values, not name)
    model_vars = add_builtin(model_vars, built_ins, model_name)
    # correcting units if subscripts are involved
    model_vars = corr_units(model_vars)

    # word analysis
    word_dict, word_list = word_analysis(model_vars, word_dict, word_list)

    # model_vars operations need to be done here

    tot, avg, funcs, f_avg, cons, unit_per = calc_avg(model_vars)
    c, d, e, f, emstocks, s_stocks, s_aux, s_flow, fs_ratio = collect_stats(model_vars)

    # finishing the documentation of current model
    # csv file contains all descriptive information about the variables in the model
    fops.save_lst_csv(model_vars, model_name, test, append=False)

    # adding the current variables to the global variable collection
    for var in model_vars:
        vardb.append(var)

    # flagging for data integrity
    # parking for debug in debug folder
    flag, code = flagging(c, f, emstocks, model_vars, built_ins)
    if flag == 'Yes':
        flag_count += 1
        fops.move_mdl_debug(mdl_file, 'flag')

    # individual operation of model is done here
    ind_end = timer()

    # adding the model stats to the stats file
    model_stat_vars.append({'Name': model_name,
                            'Variables': len(model_vars),
                            'sub_Variables': tot + cons,
                            'Constants': c['constant'],
                            'sub_Constants': cons,
                            'Auxiliaries': c['auxiliary'],
                            'sub_Auxiliaries': s_aux,
                            'Flows': c['flow'],
                            'sub_Flows': s_flow,
                            'Stocks': c['stock'],
                            'sub_Stocks': s_stocks,
                            'Flow/Stock Ratio': fs_ratio,
                            'Empty units percentage': unit_per,
                            'Table Functions': c['table function'],
                            'Subscripts': c['subscript list'],
                            'math_Simple Functions': d['simple function'],
                            'math_Complicated functions': d['complicated function'],
                            'math_Testing functions': d['testing function'],
                            'math_Incomplete equations': d['incomplete equation'],
                            'Non-constant variables': tot,
                            'Number of functions': funcs,
                            'Elements per equation': avg,
                            'Functions per equation': f_avg,
                            'Built-ins': len(built_ins),
                            'Stocks with INIT': e['yes'],
                            'Stocks without INIT': e['no'],
                            'Year': year,
                            'Flag': flag,
                            'Code': code,
                            'Time': ind_end - ind_start,
                            'Timestamp': ind_start})

    # doing the reporting for the html file
    if rep_type == 'orig':
        title = 'Original Model'
    else:
        title = 'Working Model'
    orig_df = pd.DataFrame(model_vars)
    sel = orig_df[orig_df['Number of elements'] == 1].count()['Number of elements']
    base_lst = [orig_df['Base Unit'][0], orig_df['INITIAL TIME'][0], orig_df['FINAL TIME'][0], orig_df['TIME STEP'][0]]
    cnt_lst = [len(model_vars), c['auxiliary'], c['constant'], c['flow'], c['stock'], c['table function']]
    ind_lst = [avg, unit_per, fs_ratio, f_avg, e['no'] / (e['no'] + e['yes']), sel]
    rep_tpl = (title, base_lst, cnt_lst, ind_lst)
    return model_vars, flag_count, word_dict, word_list, mis_func_list, rep_tpl


# Functions are above

# initializing the output files
dbname = 'Var_DB'
model_stat_file = 'Model_Stats'
# These are for word analysis which is currently not further developed
word_dict_file = 'word_dict'
word_list_file = 'word_list'
# These are more for testing
mis_name = 'missing_functions'
track_name = 'tracking'
err_name_adj = 'translation_errors_adjusted.txt'
err_name_std = 'translation_errors_standard.txt'

# initializing the collection lists for output files
vardb = []
track_lst = []
model_stat_vars = []
comb_func_list = []

# initilialzing the function list
# list is static and needs to be completed manually
# level of detail to be discussed
# currently no clear separation
# table function list
tbl_func_list = ['WITH LOOKUP', 'LOOKUP INVERT']
# complicated function list, roughly functions that add structure
comp_func_list = ['DELAY N', 'DELAY3', 'DELAY3I', 'DELAY FIXED', 'DELAY1', 'DELAY1I', 'DELAY MATERIAL', 'IF THEN ELSE',
                  'SAMPLE IF TRUE', 'SMOOTH', 'SMOOTHI', 'SMOOTH3', 'SMOOTH3I', 'SMOOTH N',
                  'FORECAST', 'TREND', 'NPV']
# simple functions that roughly don't add structure
simp_func_list = ['MIN', 'VMIN', 'MAX', 'VMAX', 'SIN', 'INITIAL', 'SQRT', 'ACTIVE INITIAL', 'INITIAL TIME',
                  'ZIDZ', 'XIDZ', 'SUM', 'MODULO', 'ABS', 'LN', 'SIMULTANEOUS', ':AND:', ':IMPLIES:', ':OR:',
                  'INTEGER']
# functions that are/should only be used for model testing, should probably be ignored in the testing suite
# (currently is not ignored)
test_func_list = ['RANDOM UNIFORM', 'RANDOM 0 1' 'RANDOM NORMAL', 'STEP', 'PULSE', 'PULSE TRAIN', 'RAMP', 'RND']

# documenting the init expr for all functions that have init variables
regfunc_lst = ['INTEG', 'DELAY1I', 'DELAY3I', 'SMOOTHI', 'SMOOTH3I']
nfunc_lst = ['DELAY N', 'SMOOTH N']


def full_translate(first_file=None, last_file=None):
    """
    This function runs the descriptives and translation part for all the models in the testing folder

    Output is:
    - descriptives
    - .py file of the models
    - doc file
    - statistic collections

    :param first_file: first file to be tested (index on list), defaults to None, meaning the first file in the list
    :param last_file: last file to be tested (index on list), defaults to None, meaning the last file in the list
    :return: time elapsed for the entire test as float
    """

    # initializing the counts
    model_count = 0
    flag_count = 0
    err_adj = 0
    err_std = 0

    # explorative collection for word analysis
    word_dict = {}
    word_list = []
    # missing function list
    mis_func_list = []

    total_start = timer()

    # selecting the files
    files = fops.load_files('mdl')

    for mdl_file in files[first_file:last_file]:
        print('Translating', mdl_file)
        model_count += 1
        model_name = mdl_file.split('.')[0]
        fops.output_folder(model_name, active='doc')
        # doc file contains all the information needed for further steps in the testing battery
        doc_name = mdl_file.replace('.mdl', '_doc')
        # ind_track is to keep an overview of the created documents that are necessary for next steps
        ind_track = [mdl_file, 'no', 'no', 'no']
        err_rep_adj = True
        err_rep_std = True
        # adjusted translate for descriptives
        if err_rep_adj:
            try:
                model_vars, flag_count, word_dict, word_list, mis_func_list, rep_tpl = \
                    descriptives(mdl_file, flag_count, word_dict, word_list, mis_func_list)
                ind_track[1] = 'yes'
                doc_vars = pd.DataFrame(model_vars)
            except Exception as e:
                fops.write_error_file(model_name, e, err_name_adj)
                err_adj += 1
                rep_tpl = ('Working Model', [], [], [])
                model_vars = []
                doc_vars = pd.DataFrame(model_vars)
        else:
            model_vars, flag_count, word_dict, word_list, mis_func_list, rep_tpl = \
                descriptives(mdl_file, flag_count, word_dict, word_list, mis_func_list)
            ind_track[1] = 'yes'
            doc_vars = pd.DataFrame(model_vars)
        fops.rep_work(mdl_file.rsplit('.', 1)[0], rep_tpl)
        # full pysd translation
        if err_rep_std:
            try:
                # also create the .py file for good measure
                model = pysd.read_vensim(os.path.join(source_folder, mdl_file))
                ind_track[2] = 'yes'
            except Exception as e:
                fops.write_error_file(model_name, e, err_name_std)
                err_std += 1
                fops.move_mdl_debug(mdl_file, 'trans_debug')
        else:
            model = pysd.read_vensim(os.path.join(source_folder, mdl_file))
            ind_track[2] = 'yes'
        # creation of docfile
        try:
            fulldoc = doc(doc_name, doc_vars, model.doc())
            ind_track[3] = 'yes'
            const = fulldoc.loc[fulldoc['type'] == 'constant'].reset_index()
            const = const[['Py Name', 'Real Name', 'Unit', 'expr']]
            const['fix value'] = 'NA'
            const['global minimum'] = 'NA'
            const['global maximum'] = 'NA'
            equi_name = mdl_file.replace('.mdl', '_equi')
            fops.save_csv(const, equi_name, test)
        except NameError:
            pass
        except ValueError:
            ind_track[3] = 'EMPTY'
        mdl_err_lst = []
        model_df = pd.DataFrame(model_vars)
        for i, row in model_df.iterrows():
            if row['unit'] is None:
                mdl_err_lst.append(('doc', 'Doc Error', 'No Unit Set', row['name'], '', ''))
        if len(mdl_err_lst) > 0:
            fops.init_run_error_file(mdl_err_lst, 'error_file')
        track_lst.append(ind_track)

    # transferring the tracking file to .csv
    fops.save_lst_csv(track_lst, track_name, 'report', columns=['Name', 'DB', 'python', 'doc'], append=False)

    total_end = timer()
    total_elapsed = (total_end - total_start) / 60

    print('Time elapsed for descriptive tests: ', total_elapsed, 'Minutes')
    print('pySD compatible: ', model_count - err_std)
    return total_elapsed


def create_report(first_file=None, last_file=None):
    """
    This function collects the stats and produces the reports, this is done before the models are run through
    the PySD helper

    Output is:
    - descriptives
    - statistic collections

    :param first_file: first file to be tested (index on list), defaults to None, meaning the first file in the list
    :param last_file: last file to be tested (index on list), defaults to None, meaning the last file in the list
    :return: time elapsed for the entire test as float
    """

    # initializing the counts
    model_count = 0
    flag_count = 0
    err_adj = 0
    err_std = 0

    # explorative collection for word analysis
    word_dict = {}
    word_list = []
    # missing function list
    mis_func_list = []

    total_start = timer()

    # selecting the files
    files = fops.load_files('mdl')

    for mdl_file in files[first_file:last_file]:
        print('Gathering data for', mdl_file)
        model_count += 1
        model_name = mdl_file.split('.')[0]
        fops.output_folder(model_name, active='doc')
        # ind_track is to keep an overview of the created documents that are necessary for next steps
        err_rep_adj = True
        # adjusted translate for descriptives
        if err_rep_adj:
            try:
                model_vars, flag_count, word_dict, word_list, mis_func_list, rep_tpl = \
                    descriptives(mdl_file, flag_count, word_dict, word_list, mis_func_list, 'orig')
            except Exception as e:
                fops.write_error_file(model_name, e, err_name_adj)
                err_adj += 1
                rep_tpl = ('Original Model', [], [], [])
        else:
            model_vars, flag_count, word_dict, word_list, mis_func_list, rep_tpl = \
                descriptives(mdl_file, flag_count, word_dict, word_list, mis_func_list, 'orig')
        fops.rep_orig(mdl_file.rsplit('.', 1)[0], rep_tpl)

    # transferring the general DB to .csv
    fops.save_lst_csv(vardb, dbname, 'report', append=False)
    # transferring the missing function list to .csv
    fops.save_lst_csv(mis_func_list, mis_name, 'report', append=False)
    # transferring the model statistics to .csv
    fops.save_lst_csv(model_stat_vars, model_stat_file, 'report', append=False)
    # transferring the word list to .csv
    fops.save_lst_csv(word_list, word_list_file, 'report', append=False)

    word_df = pd.DataFrame(word_dict, index=['count'])
    word_df = word_df.T
    # transferring the word dict to .csv
    fops.save_csv(word_df, word_dict_file, 'report')

    total_end = timer()
    total_elapsed = (total_end - total_start) / 60

    print('Time elapsed for descriptive tests: ', total_elapsed, 'Minutes')
    print('pySD compatible: ', model_count - err_std)
    return total_elapsed
