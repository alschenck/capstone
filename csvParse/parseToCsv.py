import re
import os

import openpyxl
import pandas as pd
import xlrd
import json

_source_dir_ = r'C:\dev\download'
_out_dir_ = r'C:\dev\download\csvs'
_config_file_ = r'C:\dev\download\file_configs.json'
file_map = None

with open(_config_file_, 'r') as j:
    file_map = json.loads(j.read())

# file_map = {'budget-data-atoz-with-rollups': {'file_pattern': 'budget-data-atoz-with-rollups',
#                                               'dest_file_name': 'budget-data-atoz-with-rollups'}
#     , 'admin_salary': {'file_pattern': 'admin_salary', 'dest_file_name': 'admin_salary','xls':{'sheet_ind':2}, 'xlsx':{'sheet_ind':2,'skip_rows':3}}
#     , 'teacher_salary': {'file_pattern': 'teacher_salary', 'dest_file_name': 'teacher_salary'}
#     , 'ada_adm': {'file_pattern': 'ada_adm', 'dest_file_name': 'ada_adm'}
#     , 'absenteeism_certified': {'file_pattern': 'absenteeism_certified', 'dest_file_name': 'absenteeism_certified'}
#     , 'act_graduates_certified': {'file_pattern': 'act_graduates_certified',
#                                   'dest_file_name': 'act_graduates_certified'}
#     , 'act_statewide_certified': {'file_pattern': 'act_statewide_certified',
#                                   'dest_file_name': 'act_statewide_certified'}
#     , 'agency_certified': {'file_pattern': 'agency_certified', 'dest_file_name': 'agency_certified'}
#     , 'ap_certified': {'file_pattern': 'ap_certified', 'dest_file_name': 'ap_certified'}
#     , 'aspire_statewide_certified': {'file_pattern': 'aspire_statewide_certified',
#                                      'dest_file_name': 'aspire_statewide_certified'}
#     , 'attendance_certified': {'file_pattern': 'attendance_certified', 'dest_file_name': 'attendance_certified'}
#     , 'community_activities_certified': {'file_pattern': 'community_activities_certified',
#                                          'dest_file_name': 'community_activities_certified'}
#     , 'discipline_actions_certified': {'file_pattern': 'discipline_actions_certified',
#                                        'dest_file_name': 'discipline_actions_certified'}
#     , 'discipline_incidents_certified': {'file_pattern': 'discipline_incidents_certified',
#                                          'dest_file_name': 'discipline_incidents_certified'}
#     , 'dropouts_certified': {'file_pattern': 'dropouts_certified', 'dest_file_name': 'dropouts_certified'}
#     , 'enrollment_certified': {'file_pattern': 'enrollment_certified', 'dest_file_name': 'enrollment_certified'}
#     , 'extra_curricular_activities_certified': {'file_pattern': 'extra_curricular_activities_certified',
#                                                 'dest_file_name': 'extra_curricular_activities_certified'}
#     , 'forward_certified': {'file_pattern': 'forward_certified', 'dest_file_name': 'forward_certified'}
#     , 'graduation_requirements_certified': {'file_pattern': 'graduation_requirements_certified',
#                                             'dest_file_name': 'graduation_requirements_certified'}
#     , 'habitual_truancy_certified': {'file_pattern': 'habitual_truancy_certified',
#                                      'dest_file_name': 'habitual_truancy_certified'}
#     , 'hs_completion_certified': {'file_pattern': 'hs_completion_certified',
#                                   'dest_file_name': 'hs_completion_certified'}
#     , 'hs_completion_legacy_rates': {'file_pattern': 'hs_completion_legacy_rates',
#                                      'dest_file_name': 'hs_completion_legacy_rates'}
#     , 'hs_completion_legacy_rates_certified': {'file_pattern': 'hs_completion_legacy_rates_certified',
#                                                'dest_file_name': 'hs_completion_legacy_rates_certified'}
#     , 'native_language_certified': {'file_pattern': 'native_language_certified',
#                                     'dest_file_name': 'native_language_certified'}
#     , 'postgrad_plans_certified': {'file_pattern': 'postgrad_plans_certified',
#                                    'dest_file_name': 'postgrad_plans_certified'}
#     , 'postsecondary_enrollment_current': {'file_pattern': 'postsecondary_enrollment_current',
#                                            'dest_file_name': 'postsecondary_enrollment_current'}
#     , 'private_enrollment_by_district_by_school_by_gender': {
#         'file_pattern': 'private_enrollment_by_district_by_school_by_gender',
#         'dest_file_name': 'private_enrollment_by_district_by_school_by_gender'}
#     , 'private_enrollment_by_district_by_school_by_grade': {
#         'file_pattern': 'private_enrollment_by_district_by_school_by_grade',
#         'dest_file_name': 'private_enrollment_by_district_by_school_by_grade'}
#     , 'private_enrollment_statewide_by_grade': {'file_pattern': 'private_enrollment_statewide_by_grade',
#                                                 'dest_file_name': 'private_enrollment_statewide_by_grade'}
#     , 'private_school_graduates': {'file_pattern': 'private_school_graduates',
#                                    'dest_file_name': 'private_school_graduates'}
#     , 'retention_certified': {'file_pattern': 'retention_certified', 'dest_file_name': 'retention_certified'}
#     , 'wsas_certified': {'file_pattern': 'wsas_certified', 'dest_file_name': 'wsas_certified'}
#     , 'private_enrollment_master': {'file_pattern': 'private_enrollment_master',
#                                     'dest_file_name': 'private_enrollment_master'}
#             }


def parse_to_csv(file):
    file_parts = file.split(".")

    if len(file_parts) < 2:
        # not valid
        return

    ext = file_parts[-1].lower()
    year = get_year(file)
    conf = get_file_obj(file)

    if year == None:
        print("error parsing year from filename : {}".format(file))
        return

    print('processing {}'.format(file))

    if ext == "csv":
        # todo
        file_data = pd.read_csv(file)
        file_name_out = ''
        if re.match(".*layout.*",file):
            file_name_out = _out_dir_ + "\\" + conf['dest_file_name'] + "_layout_" + year + ".csv"
        else:
            file_name_out = _out_dir_ + "\\" + conf['dest_file_name'] + "_" + year + ".csv"
            file_data.to_csv(file_name_out)
        return

    if ext == "xlsx":
        # todo
        bk = openpyxl.load_workbook(file)
        sheet_name = bk.get_sheet_names()[conf['xlsx']['sheet_ind']]
        skip_rows = max(0,conf['xlsx']['skip_rows'])
        file_data = pd.read_excel(file,sheet_name=sheet_name, engine="openpyxl", skiprows = skip_rows)
        file_data.to_csv(_out_dir_ + "\\" + conf['dest_file_name']+ "_" + year +".csv")
        return

    if ext == "xls":
        # todo
        # return
        bk = xlrd.open_workbook(file)
        sheet_name = bk.sheet_names()[conf['xls']['sheet_ind']]
        skip_rows = max(0, conf['xls']['skip_rows'])
        try:
            file_data = pd.read_excel(file,sheet_name=sheet_name, skiprows=skip_rows)
        except:
            file_data = pd.read_excel(file)
        file_data.to_csv(_out_dir_ + "\\" + conf['dest_file_name']+ "_" + year +".csv")
        return

    else:
        print("unknown file type: {}".format(file))
        return


def get_all_files(_source_dir_):
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(_source_dir_) for f in filenames if
            os.path.splitext(f)[1].lower() in ['.csv', '.xlsx', '.xls']]


def get_year(filename):
    try:
        reg = re.compile(".*(\d{4}).*")
        return reg.match(filename).group(1)
    except:
        print(Exception)

def get_file_obj(filename):
    for k in file_map.keys():
        conf = file_map[k]
        if re.match(".*"+conf['file_pattern']+".*",filename.lower()):
            return file_map[k]
    print("could not find config object for: {}".format(filename))

def run(file_name=None):
    for file in get_all_files(_source_dir_):
        if file_name is not None:
            if file.lower().find(file_name) > -1:
                try:
                    parse_to_csv(file)
                except:
                    print(Exception)

if __name__ == '__main__':
    # run()
    run('teacher_sal')
