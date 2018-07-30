"""
the report.py file contains the html report writing code

there are several issues with it:

- the use of div() is probably suboptimal
- every addition receives its own <head>, which is ignored in the html code but isn't nice

todo:
- remove knockout reporting

version 0.2
30.07.18/sk

"""

import dominate
import os
from dominate.tags import *
from dominate.util import raw
from configparser import ConfigParser


class Report:
    """
    the report class writes the html file for each model

    the css styles are taken from SDM doc for compatibility

    html code could be improved massively
    """
    def __init__(self, folder, file):
        self.folder = folder
        self.model_name = file
        # this should not be necessary anymore 30.07.18/sk
        #self.indicators = [1, 2, 3, 4, 5, 6]
        #self.names = ['a', 'b', 'c', 'd', 'e', 'f']
        self.report = dominate.document(title='Summary report for %s' % file)
        # this is just for testing
        self.tests = ['Original Model', 'PySD Helper', 'Working Model', 'Equilibrium', 'Monte Carlo', 'Sensitivity',
                      'Extreme', 'Time Step', 'Knockout', 'Horizon', 'Switches', 'Errors']
        self.style_table = """align='center'"""
        self.cf = ConfigParser()
        self.cf.read(os.path.join(os.path.split(folder)[0], '_config', 'settings.ini'))
        self.graph_height = self.cf['report'].get('rep_gh', fallback='120')
        self.graph_width = self.cf['report'].get('rep_gw', fallback='200')
        self.model_height = self.cf['report'].get('rep_mh', fallback='150')
        self.model_width = self.cf['report'].get('rep_mw', fallback='150')

    def save_report(self):
        """
        save report is used to save the written html elements to the html file

        one problem for this is that each new section gets its own head, etc...
        """
        with open(os.path.join(self.folder, '%s.html' % self.model_name.rsplit('.', 1)[0]), 'a', encoding='utf-8') as f:
            f.write(self.report.render())

    def set_styles(self):
        """
        when the report is initialized, the styles are set

        the styles are taken from SDM doc for compatibility
        """
        with self.report.head:
            style("""
            table.nolines {
            border-width: 3px;
            border-spacing: 0px;
            border-collapse: collapse;
            border-style: none;
            border-color: red;
            background-color: White;
            }
            table.nolines th        {
            font-size: smaller;
            background-color:blue;
            color:White;
            border-style: none;
            border-width: 0px;
            }
            table.nolines td {
            border-style: none;
            vertical-align:top;
            font-size:smaller;
            border-width: 3px;
            padding: 0px 0px 0px 0px;
            border-color: green;
            background-color: green;
            text-align:left;
            }
            table.results {
            border-width: 1px;
            border-spacing: 0px;
            border-collapse: collapse;
            border-style: solid;
            border-color: gray;
            background-color: White;
            }
            table.results th        {
            font-size: smaller;
            background-color:blue;
            color:White;
            border-style: solid;
            border-width: 1px;
            }
            table.results td {
            border-style: solid;
            vertical-align:top;
            font-size:smaller;
            border-width: 1px;
            padding: 0px 5px 0px 5px;
            border-color: gray;
            background-color: White;
            text-align:left;
            }
            table.sample {
            min-width: 400px;
            border-width: 1px;
            border-spacing: 0;
            border-style: outset;
            border-color: gray;
            border-collapse: collapse;
            }
            table.sample td {
            font-size:smaller;
            border-width: 1px;
            padding: 0px 5px 0px 5px;
            border-style: solid;
            border-color: gray;
            text-align:left;
            vertical-align:text-top;
            }
            table.sampleborder {
            min-width: 400px;
            border-width: 1px;
            border-spacing: 0;
            border-style: outset;
            border-color: gray;
            border-collapse: collapse;
            }
            table.sampleborder tr td:nth-child(4) { font-family:Consolas; }
            table.sampleborder tr td:nth-child(5) { font-family:Consolas; }
            table.sampleborder tr td:nth-child(6) { font-family:Consolas; }
            table.sampleborder tr td:nth-child(7) { font-family:Consolas; }
            table.sampleborder tr td:nth-child(8) { font-family:Consolas; }
            table.sampleborder tr td:nth-child(9) { font-family:Consolas; }
            table.sampleborder tr td:nth-child(10) { font-family:Consolas; }
            table.sampleborder tr td:nth-child(11) { font-family:Consolas; }
            table.sampleborder tr td:nth-child(12) { font-family:Consolas; }
            table.sampleborder tr td:nth-child(13) { font-family:Consolas; }
            table.sampleborder td {
            font-size:smaller;
            border-width: 1px;
            padding: 0px 5px 0px 5px;
            border-style: solid;
            border-color: gray;
            text-align:left;
            vertical-align:text-top;
            }
            table.typeLegend {
            border-width: 1px;
            border-spacing: 0;
            border-style: outset;
            border-color: gray;
            border-collapse: collapse;
            background-color: white;
            }
            table.typeLegend td {
            font-size:smaller;
            border-width: 1px;
            padding: 0px 5px 0px 5px;
            border-style: solid;
            padding: 0px 5px 0px 5px;
            border-style: solid;
            border-color: gray;
            background-color: #f8f8ff;
            text-align:left;
            }
            ul.specials li {
            color:white;
            }
            ul.specials li span {
            color:navy;
            }
            ul.footnote li {
            font-size: normal;
            }
            .verticaltext {
            writing-mode: tb-rl;
            filter: flipv fliph;
            }
            ul
            {
            list-style: disc;
            margin-top: 0;
            }""", type='text/css')

    def write_quicklinks(self):
        """
        writing the quicklinks initially, then filling them up later

        this means that there are quicklinks that might not be filled

        also new tests need to be added to the list

        could be made dependent on the the config setting to avoid empty quicklinks
        and not having to add them manually 30.07.18/sk
        """
        with self.report:
            div(h1('Report for %s' % self.model_name.rsplit('.', 1)[0]))
            with table(cls='sample'):
                th('Quicklinks', colspan='12')
                with tr():
                    for tlink in self.tests:
                        td(a(tlink, href=r'#%s' % tlink.replace(' ', '')))
            div(p('For formal requirements documentation, it is strongly recommended to use SDM-Doc which can be found',
                  a('here', href=r'https://www.systemdynamics.org/SDM-doc')))

    def write_trans(self, args):
        """
        writing the report from translation (descriptives)

        is either the original or the treated model

        :param args: tpl with arguments for the report
        """
        name, stitle, base_ind, cnt_ind, ind_ind, ms_link, doc_link = args

        if stitle == 'Original Model':
            intro = 'These stats are based on the original model prior to running it through the PySD Helper'
        else:
            intro = 'These stats are based on the working model that will be used for testing'
        base_cat = ['Base Unit', 'Initial Time', 'Final Time', 'Time Step']
        cnt_cat = ['Total Variables', 'Auxiliaries', 'Constants', 'Flows', 'Stocks', 'Table Functions']
        ind_cat = ['Elements per equation', 'Empty Units', 'Flow/Stock Ratio', 'Functions per Equation',
                   '% of Stocks without INIT', 'Equations with one element']
        with self.report:
            div(h2(a(stitle, id=stitle.replace(' ', ''))))
            div(p(intro))
            if base_ind:
                with table(cls='sample'):
                    for k, value in enumerate(base_cat):
                        with tr():
                            td(value)
                            td(base_ind[k])
                with table(cls='sample'):
                    for k, value in enumerate(cnt_cat):
                        with tr():
                            td(value)
                            td(cnt_ind[k])
                with table(cls='sample'):
                    for k, value in enumerate(ind_cat):
                        with tr():
                            td(value)
                            td(ind_ind[k])
            else:
                div(p('The model could not be translated'))
            div(a('Link to model stats', href=r'file:../%s\Model_Stats.csv' % ms_link))
            div(a('Link to the original doc file', href=r'file:./%s\doc\%s.csv' % (doc_link, name)))

    def write_helper(self, args):
        """
        writes the report of helper activities

        :param args: tpl with input for the report
        """
        stitle, cnt_lst, dfixed_lst, dinf_lst, dmat_lst, olink, nlink = args
        if not dfixed_lst:
            dfixed_lst.append('None')
        if not dinf_lst:
            dfixed_lst.append('None')
        if not dmat_lst:
            dfixed_lst.append('None')
        cnt_cat = ['# of unchangeable variables changed', '# of single quotes removed', '# of data sources replaced',
                   '# of DELAY FIXED changed', '# of DELAY INFORMATION changed', '# of DELAY MATERIAL changed',
                   '# of RANDOM 0 1 changed', '# of init variables added']
        with self.report:
            div(h2(a(stitle, id=stitle.replace(' ', ''))))
            div(p('These are the elements changed by the PySD Helper'))
            div(p('The PySD Helper might also have deleted all the comments'))
            with table(cls='results'):
                for k, value in enumerate(cnt_cat):
                    with tr():
                        td(value)
                        td(cnt_lst[k])
            div(p('The following variables have had their DELAY FIXED changed:'))
            with table(cls='results'):
                for value in dfixed_lst:
                    with tr():
                        td(value)
            div(p('The following variables have had their DELAY INFORMATION changed:'))
            with table(cls='results'):
                for value in dinf_lst:
                    with tr():
                        td(value)
            div(p('The following variables have had their DELAY MATERIAL changed:'))
            with table(cls='results'):
                for value in dmat_lst:
                    with tr():
                        td(value)
            div(a('Link to the old model', href=r'file:./%s' % olink))
            div(a('Link to the new model', href=r'file:./%s' % nlink))

    def write_equi(self, args):
        """
        writing the report for the equilibrium test

        :param args: tpl with the arguments for the report
        """
        stitle, res, value, sum_tbl, dlink = args
        summary_table = sum_tbl.to_html(classes='results', bold_rows=False)
        with self.report:
            div(h2(a(stitle, id=stitle.replace(' ', ''))))
        if res == 'NE':
            with self.report:
                div(p('No equilibrium was found'))
        elif res == 'BE':
            with self.report:
                div(p('The equilibrium found set all flows to 0'))
        else:
            with self.report:
                with table(cls='results'):
                    th('Variable')
                    th('Equi Value')
                    for k, row in value.iterrows():
                        with tr():
                            td(row['Real Name'])
                            td(row['equi'])
        with self.report:
            if not sum_tbl.empty:
                div(p('The following equilibrium conditions have been found in the base and equi run:'))
                raw(summary_table)
            else:
                div(p('No equilibrium conditions have been found in the base or equi run'))
            div(p('The timeline below shows the equilibrium condition occurences for all stocks and the model'))
            a(img(src=r'file:./%s\timeline_equi_base.png' % dlink, height=self.graph_height, width=self.graph_width),
              href=r'file:./%s\timeline_equi_base.png' % dlink)
            div(h3('The equilibrium model'))
            a(img(src=r'file:./%s\equi.gv.svg' % dlink, height=self.model_height, width=self.model_width),
              href=r'file:./%s\equi.gv.svg' % dlink)
            div(a('Link to the equilibrium folder', href=r'file:./%s' % dlink))

    def write_mc(self, args):
        """
        write report for the monte carlo test

        :param args: tpl with input for the report
        """
        stitle, constants, sp, dlink = args
        with self.report:
            div(h2(a(stitle, id=stitle.replace(' ', ''))))
            div(p('These are the resulting models from the Monte Carlo test'))
            div(p('Range tested is -%s to +%s' % (sp, sp)))
            with table(cls='results'):
                th('Variable')
                th('Model')
                for constant in constants:
                    with tr():
                        td(constant)
                        td(a(img(src=r'file:./%s\%s.gv.svg' % (dlink, constant), height=self.model_height,
                                 width=self.model_width)
                             , href=r'file:./%s\%s.gv.svg' % (dlink, constant)))
            div(a('Link to the monte carlo folder', href=r'file:./%s' % dlink))

    def write_sens(self, args):
        """
        write the report for the sensitivity test

        :param args: tpl with the input for the report
        """
        stitle, constants, sp, intervals, exo_its, endo_its, dlink = args
        sp_lst = [sp, -1 * sp]
        # clustermap are currently kicked out 21.06.18/sk
        # map_lst = ['heatmap', 'clustermap']
        map_lst = ['heatmap']
        model_lst = ['endo_run', 'endo_sens']
        with self.report:
            div(h2(a(stitle, id=stitle.replace(' ', ''))))
            div(p('These are the resulting graphs from the sensitivity test viewed from the exogenous perspective'))
            div(p('Range tested is -%s to +%s' % (sp, sp)))
            div(p('The run graph shows the behavior of all endogenous variables in the model for the sensitivity run.'))
            div(p('The normalized graph shows the behavior of all endogenous variables divided by the value of t=0.'))
            div(p('The sensititivity graph shows the behavior of all endogenous variables divided by the base run.'))
            if exo_its == 0:
                with table(cls='results'):
                    th('')
                    th('positive sensitivity', colspan='3')
                    th('negative sensitivity', colspan='3')
                    with tr():
                        td('Variable')
                        td('run graph')
                        td('normalized graph')
                        td('sensitivity graph')
                        td('run graph')
                        td('normalized graph')
                        td('sensitivity graph')
                    for constant in constants:
                        with tr():
                            td(constant)
                            for k in range(2):
                                td(a(img(src=r'file:./%s\run_%s_%s.png' % (dlink, constant, sp_lst[k]),
                                         height=self.graph_height,
                                         width=self.graph_width),
                                     href=r'file:./%s\run_%s_%s.png' % (dlink, constant, sp_lst[k])))
                                td(a(img(src=r'file:./%s\norm_%s_%s.png' % (dlink, constant, sp_lst[k]),
                                         height=self.graph_height,
                                         width=self.graph_width),
                                     href=r'file:./%s\norm_%s_%s.png' % (dlink, constant, sp_lst[k])))
                                td(a(img(src=r'file:./%s\exo_sens_%s_%s.png' % (dlink, constant, sp_lst[k]),
                                         height=self.graph_height,
                                         width=self.graph_width),
                                     href=r'file:./%s\exo_sens_%s_%s.png' % (dlink, constant, sp_lst[k])))
            else:
                for it in range(exo_its):
                    div(p('Group %s' % (it + 1)))
                    with table(cls='results'):
                        th('')
                        th('positive sensitivity', colspan='3')
                        th('negative sensitivity', colspan='3')
                        with tr():
                            td('Variable')
                            td('Run graph')
                            td('Normalized graph')
                            td('Sensitivity graph')
                            td('Run graph')
                            td('Normalized graph')
                            td('Sensitivity graph')
                        for constant in constants:
                            with tr():
                                td(constant)
                                for k in range(2):
                                    td(a(
                                        img(src=r'file:./%s\run_%s_%s_%s.png' % (dlink, constant, sp_lst[k], it),
                                            height=self.graph_height, width=self.graph_width),
                                        href=r'file:./%s\run_%s_%s_%s.png' % (dlink, constant, sp_lst[k], it)))
                                    td(a(
                                        img(src=r'file:./%s\norm_%s_%s_%s.png' % (dlink, constant, sp_lst[k], it),
                                            height=self.graph_height, width=self.graph_width),
                                        href=r'file:./%s\norm_%s_%s_%s.png' % (dlink, constant, sp_lst[k], it)))
                                    td(a(
                                        img(src=r'file:./%s\exo_sens_%s_%s_%s.png' % (dlink, constant, sp_lst[k], it),
                                            height=self.graph_height, width=self.graph_width),
                                        href=r'file:./%s\exo_sens_%s_%s_%s.png' % (dlink, constant, sp_lst[k], it)))
            div(p('These are the resulting graphs from the sensitivity test viewed from the time perspective'))
            with table(cls='results'):
                th('')
                for interval in intervals:
                    th(interval)
                for hmap in map_lst:
                    for sp in sp_lst:
                        with tr():
                            if sp > 0:
                                name = 'positive sensitivity'
                            else:
                                name = 'negative sensitivity'
                            td('%s, %s' % (hmap, name))
                            for interval in intervals:
                                td(a(
                                    img(src=r'file:./%s\%s_sensitivity in percent at t=%s_%s.png' % (
                                        dlink, hmap, interval, sp),
                                        height=self.graph_height, width=self.graph_width),
                                    href=r'file:./%s\%s_sensitivity in percent at t=%s_%s.png' % (
                                        dlink, hmap, interval, sp)))
            div(a('Link to the animated heatmap, positive sensitivity',
                  href=r'file:./%s\anim_heatmap_%s.mp4' % (dlink, sp_lst[0])))
            div(a('Link to the animated heatmap, negative sensitivity',
                  href=r'file:./%s\anim_heatmap_%s.mp4' % (dlink, sp_lst[1])))
            div(p('These are the resulting models from the sensitivity test viewed from the endogenous perspective'))
            if endo_its == 0:
                with table(cls='results'):
                    th('Model Type')
                    th('Model')
                    for mtype in model_lst:
                        for sp in sp_lst:
                            if sp > 0:
                                name = 'positive_sensitivity'
                            else:
                                name = 'negative_sensitivity'
                            with tr():
                                td('%s, %s' % (mtype, name))
                                td(a(
                                    img(src=r'file:./%s\%s_%s.gv.svg' % (dlink, mtype, name), height=self.model_height,
                                        width=self.model_width), href=r'file:./%s\%s_%s.gv.svg' % (dlink, mtype, name)))
            else:
                with table(cls='results'):
                    th('Model Type')
                    for it in range(endo_its):
                        th('Model, group %s' % it)
                    for mtype in model_lst:
                        for sp in sp_lst:
                            if sp > 0:
                                name = 'positive_sensitivity'
                            else:
                                name = 'negative_sensitivity'
                            with tr():
                                for it in range(endo_its):
                                    td('%s, %s' % (mtype, name))
                                    td(a(img(src=r'file:./%s\%s_%s_%s.gv.svg' % (dlink, mtype, name, it),
                                             height=self.model_height,
                                             width=self.model_width)
                                         , href=r'file:./%s\%s_%s_%s.gv.svg' % (dlink, mtype, name, it)))
            div(a('Link to the sensitivity folder', href=r'file:./%s' % dlink))

    def write_ext(self, args):
        """
        write the report for the extreme condition test

        :param args: tpl with input for the report
        """
        stitle, max_mult, endo_its, tbl_lst, tbl_errors, flagged, values, tbl_flagged, dlink = args
        flag_tbl = flagged.to_html(classes='results', bold_rows=False, justify='center')
        tbl_flag_tbl = tbl_flagged.to_html(classes='results', bold_rows=False, justify='center')
        values_tbl = values.to_html(classes='results', bold_rows=False, justify='center')
        ext_lst = [0, max_mult]
        with self.report:
            div(h2(a(stitle, id=stitle.replace(' ', ''))))
            div(p('The following values were used for the parameters in the extreme condition test:'))
            raw(values_tbl)
            div(p('These are the resulting models from the extreme condition test with parameters multiplied'))
            if endo_its == 0:
                with table(cls='results'):
                    th('Model Type')
                    th('Model')
                    for ext in ext_lst:
                        with tr():
                            td('Multiplied by %s' % ext)
                            td(a(img(src=r'file:./%s\mult%s.gv.svg' % (dlink, ext), height=self.model_height,
                                     width=self.model_width)
                                 , href=r'file:./%s\mult%s.gv.svg' % (dlink, ext)))
            else:
                with table(cls='results'):
                    for it in range(endo_its):
                        th('Model Type')
                        th('Model, group %s' % it)
                    for ext in ext_lst:
                        with tr():
                            for it in range(endo_its):
                                td('Multiplied by %s' % ext)
                                td(a(
                                    img(src=r'file:./%s\mult%s_%s.gv.svg' % (dlink, ext, it), height=self.model_height,
                                        width=self.model_width),
                                    href=r'file:./%s\mult%s_%s.gv.svg' % (dlink, ext, it)))
            div(p('These are the resulting models from the extreme condition test for extreme values in tables'))
            with table(cls='results'):
                th('Table')
                th('Model')
                for tbl in tbl_lst:
                    with tr():
                        td(tbl[1])
                        td(a(
                            img(src=r'file:./%s\table%s.gv.svg' % (dlink, tbl[0]), height=self.model_height,
                                width=self.model_width),
                            href=r'file:./%s\table%s.gv.svg' % (dlink, tbl[0])))
            div(p('The following variables show unexpected behavior'))
            if flagged.empty:
                div(p('No variables show unexpected behavior'))
            else:
                raw(flag_tbl)
            div(p('The following tables had runs that could not be executed'))
            if tbl_flagged.empty:
                div(p('All table runs could be executed'))
            else:
                raw(tbl_flag_tbl)
            div(p('The following formulation errors for the tables in the model have been found'))
            if len(tbl_errors) == 0:
                div(p('No formulation errors have been found'))
            else:
                with table(cls='results'):
                    th('Table')
                    th('Error')
                    for error in tbl_errors:
                        with tr():
                            td(error[3])
                            td(error[1])
            div(a('Link to the extreme condition folder', href=r'file:./%s' % dlink))

    def write_tstep(self, args):
        """
        write the report for the time step test

        :param args: tpl with input for the report
        """
        stitle, tstep, dlink = args
        with self.report:
            div(h2(a(stitle, id=stitle.replace(' ', ''))))
            div(p('Based on the time step test this is the optimal timestep'))
            with table(cls='results'):
                th('Current Time Step')
                th('Optimal Time Step')
                th('Time Step result')
                with tr():
                    td(tstep[1])
                    td(tstep[2])
                    td(tstep[3])
            div(h3('The time step model'))
            a(img(src=r'file:./%s\timestep.gv.svg' % dlink, height=self.model_height, width=self.model_width),
              href=r'file:./%s\timestep.gv.svg' % dlink)
            div(a('Link to the time step folder', href=r'file:./%s' % dlink))

    def write_ko(self, args):
        """
        write the report for the knockout test

        :param args: tpl with input for the report
        """
        stitle, ko_lst, dlink = args
        with self.report:
            div(h2(a(stitle, id=stitle.replace(' ', ''))))
            div(p('These are the modes from the knockout test'))
            with table(cls='results'):
                th('Knocked out variable')
                th('Model')
                for k_var in ko_lst:
                    with tr():
                        td(k_var)
                        td(a(
                            img(src=r'file:./%s\%s.gv.svg' % (dlink, k_var), height=self.model_height,
                                width=self.model_width),
                            href=r'file:./%s\%s.gv.svg' % (dlink, k_var)))
            div(a('Link to the knockout folder', href=r'file:./%s' % dlink))

    def write_hori(self, args):
        """
        write the report for the horizon test

        :param args: tpl with input for the report
        """
        stitle, dlink = args
        with self.report:
            div(h2(a(stitle, id=stitle.replace(' ', ''))))
            div(h3('The horizon model'))
            a(img(src=r'file:./%s\horizon.gv.svg' % dlink, height=self.model_height, width=self.model_width),
              href=r'file:./%s\horizon.gv.svg' % dlink)
            div(a('Link to the horizon folder', href=r'file:./%s' % dlink))

    def write_swit(self, args):
        """
        write the report for the switches test

        :param args: tpl with input for the report
        """
        stitle, switches, endo_its, dlink = args
        swit_type = ['full', 'sum']
        sw_settings = switches.transpose().to_html(classes='results', bold_rows=False)
        with self.report:
            div(h2(a(stitle, id=stitle.replace(' ', ''))))
            div(p('This summarizes the switch settings'))
            if switches.empty:
                div(p('There are no switches in the model'))
            else:
                raw(sw_settings)
            div(a('Link to the switch settings file', href=r'file:./%s\switch_settings.csv' % dlink))
            div(p('The full switch settings report every possible switch combination. The summarized switch settings '
                  'report the runs where each switch is activated as the only switch.'))
            if endo_its == 0:
                with table(cls='results'):
                    th('Model Type')
                    th('Model')
                    for stype in swit_type:
                        if stype == 'full':
                            name = 'full switch settings'
                        else:
                            name = 'summarized switch settings'
                        with tr():
                            td(name)
                            td(a(img(src=r'file:./%s\%s.gv.svg' % (dlink, stype), height=self.model_height,
                                     width=self.model_width)
                                 , href=r'file:./%s\%s.gv.svg' % (dlink, stype)))
            else:
                with table(cls='results'):
                    th('Model Type')
                    for it in range(endo_its):
                        th('Model, group %s' % it)
                    for stype in swit_type:
                        if stype == 'full':
                            name = 'full switch settings'
                        else:
                            name = 'summarized switch settings'
                        with tr():
                            td(name)
                            for it in range(endo_its):
                                if type == 'sum':
                                    if it == 0:
                                        td(a(img(src=r'file:./%s\%s.gv.svg' % (dlink, stype),
                                                 height=self.model_height,
                                                 width=self.model_width)
                                             , href=r'file:./%s\%s.gv.svg' % (dlink, stype)))
                                    else:
                                        pass
                                else:

                                    td(a(img(src=r'file:./%s\%s_%s.gv.svg' % (dlink, stype, it),
                                             height=self.model_height,
                                             width=self.model_width)
                                         , href=r'file:./%s\%s_%s.gv.svg' % (dlink, stype, it)))
            div(a('Link to the switches folder', href=r'file:./%s' % dlink))

    def write_errors(self, args):
        """
        write the errors to the report at the end

        called from the battery because it's the summed up errors from all tests

        :param args: tpl with the input for the report
        """
        stitle, counts, dlink = args
        if link is not None:
            err_lst = counts.to_frame(name='Errors').to_html(classes='results', bold_rows=False)
            with self.report:
                div(h2(a(stitle, id=stitle.replace(' ', ''))))
                div(p('The tests have found the following errors:'))
                raw(err_lst)
                div(a('Link to the error file', href=r'file:./%s' % dlink))
