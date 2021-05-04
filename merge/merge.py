import pandas as pd
import numpy as np
import os


_csv_path_ = f'C:\dev\download\csvs'
_csv_all_path_ = f'C:\dev\download\csvs_all'
_data_frames_ = {
    'absenteeism_certified': pd.DataFrame,
    'act_graduates_certified': pd.DataFrame,
    'act_statewide_certified': pd.DataFrame,
    # 'ada_adm': pd.DataFrame,
    'admin_salary': pd.DataFrame,
    # 'agency_certified': pd.DataFrame,
    'ap_certified': pd.DataFrame,
    # 'aspire_statewide_certified': pd.DataFrame,
    #todo
    'attendance_certified': pd.DataFrame,
    # 'budget-data-atoz-with-rollups': pd.DataFrame,
    'community_activities_certified': pd.DataFrame,
    # 'discipline_actions_certified': pd.DataFrame,
    # 'discipline_incidents_certified': pd.DataFrame,
    'dropouts_certified': pd.DataFrame,
    'enrollment_certified': pd.DataFrame,
    'extra_curricular_activities_certified': pd.DataFrame,
    # 'forward_certified': pd.DataFrame,
    'habitual_truancy_certified': pd.DataFrame,
    'hs_completion_certified': pd.DataFrame,
    # 'hs_completion_legacy_rates': pd.DataFrame,
    # 'native_language_certified': pd.DataFrame,
    #'postgrad_plans_certified': pd.DataFrame,
    'postsecondary_enrollment_current': pd.DataFrame,
    'retention_certified': pd.DataFrame,
    'teacher_salary': pd.DataFrame
    # 'wsas_certified': pd.DataFrame
}


def load_csv(f=None):
    for file in _data_frames_.keys():
        if file.lower().find(f) > -1:
            curr_files = [f for f in os.listdir(_csv_path_) if f.find(file) > -1]
            merged_file = None
            for file_name in curr_files:
                year = file_name.split('_')[-1].split('.')[0]
                curr_file = pd.read_csv(_csv_path_ + '\\' + file_name)
                curr_file['year'] = year
                curr_file.drop(columns='Unnamed: 0')
                merged_file = curr_file if merged_file is None else merged_file.append(curr_file)
                print(file_name)
            merged_file.to_csv(_csv_all_path_ + '\\' + file + '_all.csv')


def reform_csv():
    for file in _data_frames_.keys():
        print(file)
        file_pd = pd.read_csv(_csv_all_path_ + '\\' + file + '_all.csv')
        file_pd = file_pd[file_pd.year >= 2010]
        file_pd = file_pd[file_pd.year <= 2016]

        if file in ('absenteeism_certified', 'act_graduates_certified', 'act_statewide_certified','ap_certified'
                    ,'attendance_certified','community_activities_certified'
                    ,'dropouts_certified','enrollment_certified','extra_curricular_activities_certified'
                    ,'habitual_truancy_certified','hs_completion_certified','postsecondary_enrollment_current','retention_certified'):
            file_pd = file_pd[file_pd['SCHOOL_NAME'] == '[Districtwide]']
            file_pd = file_pd[file_pd['AGENCY_TYPE'] == 'School District']

        if file == 'absenteeism_certified':
            file_pd = file_pd[
                ['DISTRICT_CODE', 'GROUP_BY', 'GROUP_BY_VALUE', 'ABSENTEE_MEASURE', 'ABSENCE_RATE', 'year']]
            _data_frames_[file] = file_pd.pivot(index=['DISTRICT_CODE', 'year'],
                                                columns=['GROUP_BY', 'GROUP_BY_VALUE', 'ABSENTEE_MEASURE'],
                                                values='ABSENCE_RATE')

        if file in ['act_graduates_certified', 'act_statewide_certified']:
            file_pd = file_pd[['DISTRICT_CODE', 'GROUP_BY', 'GROUP_BY_VALUE', 'TEST_SUBJECT', 'AVERAGE_SCORE', 'year']]
            file_pd = file_pd.drop_duplicates()
            file_pd = file_pd[file_pd.AVERAGE_SCORE.isna() == False]
            file_pd = file_pd[file_pd.AVERAGE_SCORE != '*']
            _data_frames_[file] = file_pd.pivot(index=['DISTRICT_CODE', 'year'],
                                                columns=['GROUP_BY', 'GROUP_BY_VALUE', 'TEST_SUBJECT'],
                                                values='AVERAGE_SCORE')

        if file == 'admin_salary':
            prev_format = file_pd[
                ['Dist No.', 'year', 'Local Exp', 'Total Exp', 'High Degree',
                 'Pos FTE',
                 'Prorated Salary', 'Prorated Fringe'
                 ,'Hire Dist No.','Work District','Work District Code','Work Agency Code','Work Dist Code'

                 ]]

            prev_format = prev_format[prev_format['Pos FTE'] > 0]
            prev_format = prev_format.drop_duplicates()
            prev_format.loc[prev_format['Pos FTE'] >= 10,'Pos FTE'] = prev_format['Pos FTE']/100

            prev_format['Work District Num'] = pd.to_numeric(prev_format['Work District'], errors='coerce').fillna(0).astype(np.int64)
            prev_format['Work Dist Num'] = pd.to_numeric(prev_format['Work Dist Code'], errors='coerce').fillna(
                0).astype(np.int64)

            prev_format['Dist No.'] = prev_format['Dist No.'].fillna(0).astype(np.int64)+prev_format['Hire Dist No.'].fillna(0).astype(np.int64)+prev_format['Work District Code'].fillna(0).astype(np.int64)+prev_format['Work District Num'].fillna(0).astype(np.int64)+prev_format['Work Agency Code'].fillna(0).astype(np.int64)+prev_format['Work Dist Num'].fillna(0).astype(np.int64)

            prev_format['ADJUSTED_FRINGE'] = prev_format['Prorated Fringe']/prev_format['Pos FTE']
            prev_format['ADJUSTED_SAL'] = prev_format['Prorated Salary'] / prev_format['Pos FTE']
            prev_format['Dist No.'].fillna(0).astype(np.int64)
            prev_format = prev_format[['Dist No.','year','Local Exp','Total Exp','High Degree','ADJUSTED_FRINGE','ADJUSTED_SAL']]
            prev_format.drop_duplicates(subset=['Dist No.','year','Local Exp','Total Exp','High Degree'], ignore_index=True, inplace=True)


            new_format = file_pd[
                ['Work District Code', 'Work Agency Code', 'year', 'Local Experience', 'Total Experience','High Degree Desc',
                  'Position FTE',
                 'Total Salary', 'Total Fringe'
                , 'Hire Dist No.', 'Work District', 'Work Dist Code'
                ]]
            new_format = new_format[new_format['Position FTE'] > 0]
            new_format = new_format.drop_duplicates()

            new_format['Dist No.'] = new_format['Work District Code'].fillna(0).astype(np.int64) + new_format[
                'Hire Dist No.'].fillna(0).astype(np.int64)  + new_format['Work Agency Code'].fillna(0).astype(np.int64)

            # 8 = Other, 7 = Ph.D., 6 = 6-year Specialist, 5 = M.A. or M.S., 4 = B.A. or B.S#
            # M.A. or M.S.
            # Ph.D.
            # 6-year Specialist
                # B.A. or B.S.
            # Other
            # Master's Degree
                # Bachelor's Degree
            # Six Year Specialist's
            # Doctorate

            new_format = new_format.replace({
                'High Degree Desc': {
                    'Other': 8,
                    'Ph.D.': 7,
                    'Doctorate': 7,
                    '6-year Specialist': 6,
                    'Six Year Specialist\'s': 6,
                    'M.A. or M.S.': 5,
                    'Master\'s Degree': 5,
                    'Bachelor\'s Degree': 4,
                    'B.A. or B.S.':4                }
            })
            new_format = new_format[['Dist No.','year','Local Experience','Total Experience','High Degree Desc','Position FTE','Total Salary','Total Fringe']]
            new_format = new_format.groupby(['Dist No.', 'year','Local Experience','Total Experience','High Degree Desc','Total Salary','Total Fringe']).sum().reset_index()
            new_format['ADJUSTED_FRINGE'] = new_format['Total Fringe']/new_format['Position FTE']*100
            new_format['ADJUSTED_SAL'] = new_format['Total Salary']/new_format['Position FTE']*100
            new_format = new_format[['Dist No.','year','Local Experience','Total Experience','High Degree Desc','ADJUSTED_FRINGE','ADJUSTED_SAL']]

            new_format.columns = ['DISTRICT_CODE','year','local_exp','tot_exp','high_degree','adjusted_fringe','adjusted_sal']
            prev_format.columns = ['DISTRICT_CODE', 'year', 'local_exp', 'tot_exp', 'high_degree', 'adjusted_fringe',
                                  'adjusted_sal']
            prev_format['DISTRICT_CODE'] = prev_format['DISTRICT_CODE'].fillna(0).astype(np.int64)

            _data_frames_[file] = new_format.append(prev_format).copy().groupby(['DISTRICT_CODE','year']).mean()
            del new_format
            del prev_format

        if file == 'ap_certified':
            file_pd = file_pd.drop(columns=['Unnamed: 0','Unnamed: 0.1','SCHOOL_YEAR','AGENCY_TYPE','CESA','COUNTY','SCHOOL_CODE','DISTRICT_NAME','SCHOOL_NAME','EXAM_COUNT','EXAMS_3_OR_ABOVE','CHARTER_IND'])
            file_pd = file_pd.drop_duplicates()
            file_pd = file_pd[file_pd.PERCENT_3_OR_ABOVE.isna() == False]
            file_pd = file_pd[file_pd.PERCENT_3_OR_ABOVE != '*']
            _data_frames_[file] = file_pd.pivot(index=['DISTRICT_CODE', 'year'],
                                                columns=['GROUP_BY', 'GROUP_BY_VALUE', 'AP_EXAM'],
                                                values='PERCENT_3_OR_ABOVE')

        if file == 'attendance_certified':
            file_pd = file_pd[['DISTRICT_CODE', 'GROUP_BY', 'GROUP_BY_VALUE', 'ATTENDANCE_RATE', 'year']]
            file_pd = file_pd.drop_duplicates()
            file_pd = file_pd[file_pd.ATTENDANCE_RATE.isna() == False]
            file_pd = file_pd[file_pd.ATTENDANCE_RATE != '*']
            _data_frames_[file] = file_pd.pivot(index=['DISTRICT_CODE', 'year'],
                                                columns=['GROUP_BY', 'GROUP_BY_VALUE'],
                                                values='ATTENDANCE_RATE')

        if file in  ('community_activities_certified','extra_curricular_activities_certified'):
            file_pd = file_pd[['DISTRICT_CODE', 'ACTIVITY_TYPE', 'PARTICIPATION_RATE', 'year']]
            file_pd = file_pd.drop_duplicates()
            file_pd = file_pd[file_pd.PARTICIPATION_RATE.isna() == False]
            file_pd = file_pd[file_pd.PARTICIPATION_RATE != '*']
            _data_frames_[file] = file_pd.pivot(index=['DISTRICT_CODE', 'year'],
                                                columns=['ACTIVITY_TYPE'],
                                                values='PARTICIPATION_RATE')

        if file == 'dropouts_certified':
            file_pd = file_pd[['DISTRICT_CODE', 'GROUP_BY', 'GROUP_BY_VALUE', 'DROPOUT_RATE', 'year']]
            file_pd = file_pd.drop_duplicates()
            file_pd = file_pd[file_pd.DROPOUT_RATE.isna() == False]
            file_pd = file_pd[file_pd.DROPOUT_RATE != '*']
            _data_frames_[file] = file_pd.pivot(index=['DISTRICT_CODE', 'year'],
                                                columns=['GROUP_BY', 'GROUP_BY_VALUE'],
                                                values='DROPOUT_RATE')

        if file == 'enrollment_certified':
            file_pd = file_pd[['DISTRICT_CODE', 'GROUP_BY', 'GROUP_BY_VALUE', 'PERCENT_OF_GROUP', 'year']]
            file_pd = file_pd.drop_duplicates()
            file_pd = file_pd[file_pd.PERCENT_OF_GROUP.isna() == False]
            file_pd = file_pd[file_pd.PERCENT_OF_GROUP != '*']
            _data_frames_[file] = file_pd.pivot(index=['DISTRICT_CODE', 'year'],
                                                columns=['GROUP_BY', 'GROUP_BY_VALUE'],
                                                values='PERCENT_OF_GROUP')

        if file == 'habitual_truancy_certified':
            file_pd = file_pd[['DISTRICT_CODE', 'GROUP_BY', 'GROUP_BY_CATEGORY', 'TRUANCY_RATE', 'year']]
            file_pd = file_pd.drop_duplicates()
            file_pd = file_pd[file_pd.TRUANCY_RATE.isna() == False]
            file_pd = file_pd[file_pd.TRUANCY_RATE != '*']
            _data_frames_[file] = file_pd.pivot(index=['DISTRICT_CODE', 'year'],
                                                columns=['GROUP_BY', 'GROUP_BY_CATEGORY'],
                                                values='TRUANCY_RATE')

        if file == 'hs_completion_certified':
            file_pd = file_pd[['DISTRICT_CODE', 'GROUP_BY', 'GROUP_BY_VALUE','COMPLETION_STATUS','TIMEFRAME','COHORT_COUNT', 'STUDENT_COUNT', 'year']]
            file_pd = file_pd.drop_duplicates()
            file_pd = file_pd[file_pd.STUDENT_COUNT.isna() == False]
            file_pd = file_pd[file_pd.STUDENT_COUNT != '*']
            _data_frames_[file] = file_pd.pivot(index=['DISTRICT_CODE', 'year'],
                                                columns=['GROUP_BY', 'GROUP_BY_VALUE','TIMEFRAME','COMPLETION_STATUS'],
                                                values=['COHORT_COUNT', 'STUDENT_COUNT'])

        if file == 'postsecondary_enrollment_current':
            file_pd = file_pd[['DISTRICT_CODE', 'GROUP_BY', 'GROUP_BY_VALUE','INITIAL_ENROLLMENT','INSTITUTION_LOCATION','INSTITUTION_LEVEL','INSTITUTION_TYPE', 'GROUP_COUNT','STUDENT_COUNT','year']]
            file_pd = file_pd.drop_duplicates()
            file_pd = file_pd[file_pd.STUDENT_COUNT.isna() == False]
            file_pd = file_pd[file_pd.STUDENT_COUNT != '*']
            _data_frames_[file] = file_pd.pivot(index=['DISTRICT_CODE', 'year'],
                                                columns=['GROUP_BY', 'GROUP_BY_VALUE','INITIAL_ENROLLMENT','INSTITUTION_LOCATION','INSTITUTION_LEVEL','INSTITUTION_TYPE'],
                                                values=['GROUP_COUNT','STUDENT_COUNT'])

        if file == 'retention_certified':
            file_pd = file_pd[['DISTRICT_CODE', 'GROUP_BY', 'GROUP_BY_VALUE', 'RETENTION_RATE', 'year']]
            file_pd = file_pd.drop_duplicates()
            file_pd = file_pd[file_pd.RETENTION_RATE.isna() == False]
            file_pd = file_pd[file_pd.RETENTION_RATE != '*']
            _data_frames_[file] = file_pd.pivot(index=['DISTRICT_CODE', 'year'],
                                                columns=['GROUP_BY', 'GROUP_BY_VALUE'],
                                                values='RETENTION_RATE')

        if file == 'teacher_salary':
            file_pd = file_pd[['Dist No.' , 'Dist Code','District Code','Agency Code','year' ,'Low Salary','High Salary'
                ,'Average Salary','Average Fringe','Average Local Experience','Average Total Experience'
                ,'Average Salary (Sorted by)']]

            file_pd['DISTRICT_CODE'] = file_pd['Dist No.'].fillna(0).astype(np.int64) +file_pd['Dist Code'].fillna(0).astype(np.int64) + file_pd['District Code'].fillna(0).astype(np.int64)  + file_pd['Agency Code'].fillna(0).astype(np.int64)
            file_pd['AVERAGE_SALARY'] = file_pd['Average Salary'].fillna(0).astype(np.int64) + file_pd['Average Salary (Sorted by)'].fillna(0).astype(np.int64)

            file_pd = file_pd[['DISTRICT_CODE','year','Low Salary','AVERAGE_SALARY','High Salary','Average Fringe', 'Average Local Experience','Average Total Experience']]
            file_pd.columns = ['DISTRICT_CODE','year','LOW_SALARY','AVERAGE_SALARY','HIGH_SALARY','AVERAGE_FRINGE','AVG_LOCAL_EXP','AVG_TOT_EXP']


            file_pd = file_pd.drop_duplicates()
            file_pd = file_pd[file_pd.AVERAGE_SALARY.isna() == False]
            file_pd = file_pd[file_pd.AVERAGE_SALARY != '*']
            _data_frames_[file] = file_pd.set_index(['DISTRICT_CODE','year'])

def run():
    # load_csv()
    # load_csv('teacher_salary')
    reform_csv()
    merge_dfs()

def merge_dfs():
    df = pd.DataFrame
    count = 0
    for file in _data_frames_.keys():
        print(file)
        # print(_data_frames_[file].index)
        df2 = _data_frames_[file].copy()
        cols = [f for f in _data_frames_[file].columns]
        for i in range(len(cols)):
            if type(cols[i]) == type((1, 2)):
                cols[i] = tuple([file] + list(cols[i]))
            else:
                cols[i] = file + "_" + cols[i]

        df2.columns = cols
        if count == 0:
            df = df2.copy()
        else:
            df = df.merge(df2.copy(),how='outer',left_index=True,right_index=True )
        count = count +1
    df = df.filter(regex='^\([1-9].*' ,axis=0) #filter out district 0

    df.to_csv(_csv_all_path_+'\\master_list.csv')


if __name__ == '__main__':
    run()
