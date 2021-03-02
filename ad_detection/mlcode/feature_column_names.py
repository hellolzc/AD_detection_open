import os
import json

GROUP_DICT = {}


def get_filter_items(feature_group):
    """ feature_group : 选择的特征集
        duration egemaps linguistic score demographics doctor all
        返回选择的特征组合的所用特征列名
    """
    filter_items = GROUP_DICT[feature_group]
    return filter_items


def get_category(feat_name):
    """ 根据特征名识别出属于的特征组
    """
    if feat_name in GROUP_DICT['demographics']:
        return 'Demographic'
    elif feat_name in GROUP_DICT['score']:
        return 'Score'
    elif feat_name in GROUP_DICT['egemaps']:
        return 'Acoustic'
    elif feat_name in GROUP_DICT['duration']:
        return 'Duration'
    elif feat_name in GROUP_DICT['linguistic']:
        return 'Linguistic'
    # elif feat_name in cog_label_cols:
    #     return 'Cognitive'
    elif feat_name[0:4] == 'perp':
        return 'Linguistic'
    elif feat_name[0:4] in ['IS09', 'IS10', 'IS11', 'IS12', 'CPE1', 'egem']:
        return 'Acoustic'
    else:
        print('Warning: Name "%s" is undefined!' % feat_name)
        return 'Undefined'

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), './feat_col_names.json')
def update_group_dict():
    with open(CONFIG_FILE_PATH,'rt') as jsonFile:
        val = jsonFile.read()
        messageConfig = json.loads(val)
        set_cols = messageConfig["feature_set_cols"]
        for key in set_cols:
            set_cols[key] = set_cols[key].split(',')

    global GROUP_DICT
    GROUP_DICT = {
        'duration': set_cols["duration"],
        'egemaps': set_cols["egemaps"],
        # remember the perplexities are added in CV not here.
        'linguistic': set_cols["linguistic_B"] + set_cols["syntactic"],
        'score': set_cols["score"],
        'demographics': set_cols["demographics"],
        'doctor': set_cols["demographics"] + set_cols["score"],
        'all': set_cols["demographics"] + set_cols["duration"] + set_cols["egemaps"],  # + linguistic + syntactic
        'propose': set_cols["demographics"] + set_cols["duration"],
        'select': set_cols["select"],
        # 'cog': cog_label_cols,
        'test': set_cols["score"] + set_cols["duration"],
    }
    # GROUP_DICT['all_with_cog'] = GROUP_DICT['all']+GROUP_DICT['cog']
    return set_cols


update_group_dict()
