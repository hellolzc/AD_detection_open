#!/usr/bin/env python
import pandas as pd

data_label_path = '../ws_en/label/data.csv'
LABEL_DF = pd.read_csv(data_label_path, index_col=0)

def get_info(uuid):
    """ Load data.csv, and return information of spcified uuid: age,sex,group,label,education
    uuid is a string like '010-3c', '714-0c'
    Note from Pitt corpus:
    The first first 3 digits correspond to the id numbers in column A of the Data Spreadsheet. 
    The number following the dash corresponds to the visit number: 0 (baseline), 1, 2, 3, and 4.
    On the Data Spreadsheet, these numbers are entered as 1 (baseline), 2, 3, 4, and 5.
    So, 001-0.cha corresponds to visit 1 data on the spreadsheet for id #1;
     001-1.cha corresponds to visit 2 data on the spreadsheet for id #1, etc.
    """
    uuid_splited = uuid.split('-')
    participant_id = int(uuid_splited[0])
    visit_no = int(uuid_splited[1][:-1])
    
    re_dict = {
        'participant_id': participant_id,
        'visit_no': visit_no,
        'idate': LABEL_DF.loc[participant_id, 'idate'],
        'education': LABEL_DF.loc[participant_id, 'educ'],
    }
    re_dict['diagnosis'] = LABEL_DF.loc[participant_id, 'curdx%d' % (visit_no + 1)]
    if visit_no == 0:
        re_dict['mmse'] = LABEL_DF.loc[participant_id, 'mms']
    else:
        re_dict['mmse'] = LABEL_DF.loc[participant_id, 'mmse%d' % (visit_no + 1)]
    return re_dict


def get_education(uuid):
    uuid_splited = uuid.split('-')
    participant_id = int(uuid_splited[0])
    return LABEL_DF.loc[participant_id, 'educ']

def test():
    print(LABEL_DF.head())
    print(get_info('007-1c'))
    print(get_info('001-0c'))
    print(get_info('001-2c'))
    for uuid in ['010-0c', '010-1c', '010-2c', '010-3c', '013-0c','013-2c', '013-3c', '013-4c']:
        print(get_info(uuid))


def add_educ_to_summary():
    data_label_path = '../label/chat_tier_info.csv'
    df = pd.read_csv(data_label_path, index_col=0)
    df['education'] = 0
    for uuid in df.index:
        df.loc[uuid, 'education'] = get_education(uuid)
    
    df.to_csv('../label/summary_auto.csv')

if __name__ == "__main__":
    add_educ_to_summary()
