import requests

_baseUrl_ = "https://dpi.wi.gov/sites/default/files/wise/downloads/"
_dir_ = "c:\dev\download"
starting_year = 21


def getStudentData():
    files = [
        "act_graduates_certified_"
        , "act_statewide_certified_"
        , "agency_certified_"
        , "ap_certified_"
        , "aspire_statewide_certified_"
        , "absenteeism_certified_"
        , "attendance_dropouts_certified_"
        , "average_daily_attendance_"
        , "discipline_actions_certified_"
        , "discipline_incidents_certified_"
        , "enrollment_certified_"
        , "private_enrollment_"
        , "extra_curricular_activities_certified_"
        , "forward_certified_"
        , "graduation_requirements_certified_"
        , "habitual_truancy_certified_"
        , "hs_completion_certified_"
        , "hs_completion_legacy_rates_certified_"
        , "private_school_graduates_"
        , "native_language_certified_"
        , "postgrad_plans_certified_"
        , "postsecondary_enrollment_current_"
        , "retention_certified_"
        , "community_activities_certified_"
        , "all_topics_winss_"
        , "wsas_certified_"]

    for file in files:
        for i in range(20):
            file_name = file + "20{}-{}.zip".format(starting_year - i, starting_year - (i - 1))
            z = requests.get(_baseUrl_ + file_name)
            if z.ok:
                f = open(_dir_ + "\\" + file_name, "wb")
                f.write(z.content)
                f.close()


def getBudgetData():
    budget_files = [
        "https://dpi.wi.gov/sites/default/files/imce/sfs/SFSDW_CSV_DATAFILES/2017-18/2017-2018-12-Budget-Data-AtoZ-with-Rollups.csv"
        ,
        "https://dpi.wi.gov/sites/default/files/imce/sfs/SFSDW_CSV_DATAFILES/aa2016-2017/2016-2017-12-Budget-Data-AtoZ-with-Rollups.csv"
        ,
        "https://dpi.wi.gov/sites/default/files/imce/sfs/SFSDW_CSV_DATAFILES/a2015-2016/2015-2016-11-Budget-Data-AtoZ-with-Rollups.csv"
        ,
        "https://dpi.wi.gov/sites/default/files/imce/sfs/SFSDW_CSV_DATAFILES/b2014-2015/2014-2015-12-Budget-Data-AtoZ-with-Rollups.csv"
        ,
        "https://dpi.wi.gov/sites/default/files/imce/sfs/SFSDW_CSV_DATAFILES/d2013-2014/d2013-2014-01-Budget-Data-AtoZ-with-rollups.csv"
        ,
        "https://dpi.wi.gov/sites/default/files/imce/sfs/SFSDW_CSV_DATAFILES/d2012-2013/2012-2013-01-Budget-Data-AtoZ-with-rollups.csv"]

    for file in budget_files:
        file_name = file.split("/")[-1]
        z = requests.get(file)
        if z.ok:
            f = open(_dir_ + "\\" + file_name, "wb")
            f.write(z.content)
            f.close()


def getAdminData():
    admin_files = [
        (2016, "/sites/default/files/imce/cst/Copy%20of%202015-2016%20Admin%20Salary%20Report%20noWISEid.xlsx")
        , (2015, "/sites/default/files/imce/cst/2014-2015%20Admin%20Salary%20Report%20No%20WISEid.xlsx")
        , (2014, "/sites/default/files/imce/cst/xls/2013_2014_administrative_salary_report.xlsx")
        , (2013, "/sites/default/files/imce/cst/xls/2012_2013_administrative_salary_report.xlsx")
        , (2012, "/sites/default/files/imce/cst/xls/adm_sal_2012_final.xls")
        , (2011, "/sites/default/files/imce/cst/xls/adm_sal_2011_final.xls")
        , (2010, "/sites/default/files/imce/cst/xls/adm_sal_2010_final.xls")
        , (2009, "/sites/default/files/imce/cst/xls/adm_sal_2009_final.xls")
        , (2008, "/sites/default/files/imce/cst/xls/adm_sal_2008_final.xls")
        , (2007, "/sites/default/files/imce/cst/xls/adm_sal_2007_final.xls")
        , (2006, "/sites/default/files/imce/cst/xls/adm_sal_2006.xls")
        , (2005, "/sites/default/files/imce/cst/xls/admsal05.xls")
        , (2004, "/sites/default/files/imce/cst/xls/admsal04.xls")
        , (2003, "/sites/default/files/imce/cst/xls/admsal03.xls")
        , (2002, "/sites/default/files/imce/cst/xls/admsal02.xls")
        , (2001, "/sites/default/files/imce/cst/xls/admsal01.xls")
        , (2000, "/sites/default/files/imce/cst/xls/admsal00.xls")
        , (1999, "/sites/default/files/imce/cst/xls/adsal99.xls")
        , (1998, "/sites/default/files/imce/cst/xls/adsal98.xls")]
    _baseUrl_ = "https://dpi.wi.gov"
    for yr, file in admin_files:
        file_name = 'admin_salary_{}.{}'.format(yr, file.split('.')[1])
        z = requests.get(_baseUrl_ + file)
        if z.ok:
            f = open(_dir_ + "\\" + file_name, "wb")
            f.write(z.content)
            f.close()


def getTeacherData():
    teacher_files = [(2016,
                      "/sites/default/files/imce/cst/xls/2015-2016%20TASR%20-%20Average%20Teacher%20Salary%20Report%20by%20Agency.xlsx")
        , (2015,
           "/sites/default/files/imce/cst/xls/2014-2015%20TASR%20-%20Average%20Teacher%20Salary%20Report%20by%20Agency.xlsx")
        , (2014, "/sites/default/files/imce/cst/xls/tasr14.xlsx")
        , (2013, "/sites/default/files/imce/cst/xls/tasr13.xlsx")
        , (2012, "/sites/default/files/imce/cst/xls/tasr12.xls")
        , (2011, "/sites/default/files/imce/cst/xls/tasr11.xls")
        , (2010, "/sites/default/files/imce/cst/xls/tasr10.xls")
        , (2009, "/sites/default/files/imce/cst/xls/tasr09.xls")
        , (2008, "/sites/default/files/imce/cst/xls/tasr08.xls")
        , (2007, "/sites/default/files/imce/cst/xls/tasr07.xls")
        , (2006, "/sites/default/files/imce/cst/xls/tasr06.xls")
        , (2005, "/sites/default/files/imce/cst/xls/tasr05.xls")
        , (2004, "/sites/default/files/imce/cst/xls/tasr04.xls")
        , (2003, "/sites/default/files/imce/cst/xls/tasr03.xls")
        , (2002, "/sites/default/files/imce/cst/xls/tasr02.xls")
        , (2001, "/sites/default/files/imce/cst/xls/tasr01.xls")
        , (2000, "/sites/default/files/imce/cst/xls/tasr00.xls")
        , (1999, "/sites/default/files/imce/cst/xls/tasr99.xls")
        , (1998, "/sites/default/files/imce/cst/xls/tasr98.xls")]
    _baseUrl_ = "https://dpi.wi.gov"
    for yr, file in teacher_files:
        file_name = 'teacher_salary_{}.{}'.format(yr, file.split('.')[1])
        z = requests.get(_baseUrl_ + file)
        if z.ok:
            f = open(_dir_ + "\\" + file_name, "wb")
            f.write(z.content)
            f.close()


def getAllData():
    getStudentData()
    getBudgetData()
    getAdminData()
    getTeacherData()


if __name__ == '__main__':
    pass
