from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import librosa
import librosa.display


def get_csv_df(index, fname, rate=None):
    if rate is None:
        if len(index) == 6:
            rate = 1.0
        elif index[9] == '1':  # a01 0.9是慢放
            rate = 1 / 0.9
        elif index[9] == '2':  # a02 1.1是快放
            rate = 1 / 1.1
        else:
            raise ValueError('Wrong index: %s' % index)
    # fname = get_csv_fname(index)
    print('Read File: %s' % fname )
    df = pd.read_csv(fname, sep='\t')
    df['start_time'] *= rate
    df['end_time'] *= rate
    return df

def plot_wav_attention(index: str, wav_fname: str, heatmaps: List, truncate_time: float=102.4, scale: float=None):
    # index = get_index_gt_pred(test_no)[0]
    # wav_fname = get_wav_fname(index)
    data, sampling_rate = librosa.load(wav_fname, sr=None, duration=truncate_time)
    
    if scale is None:
        max_abs_value = 0
        for i, heatmap in enumerate(heatmaps):
            max_abs_value_i = np.max(np.abs(heatmap))
            if max_abs_value_i > max_abs_value:
                max_abs_value = max_abs_value_i
        scale = 1.0/max_abs_value
        print("[INFO @ %s]"%__name__, 'max_abs_value:', max_abs_value)

    # plt.title(wav_fname + '     ' + str(get_index_gt_pred(test_no)))
    for i, heatmap in enumerate(heatmaps):
        x = np.linspace(0, truncate_time, num=len(heatmap))
        plt.plot(x, heatmap * scale, color=f'C{i+1}', linestyle='-')
    librosa.display.waveplot(data, sr=sampling_rate)


def _find_break_pos(txt, break_pos, allow_shift=2):
    """ 简单的函数，用来找合适的断行位置
    allow_shift << break_len
    """
    txt_len = len(txt)
    if txt[break_pos] == ' ':
        return 0, True
    for i in range(allow_shift):
        for shift in (i+1, -i-1):
            new_break_pos = break_pos + shift
            if new_break_pos >= txt_len or txt[new_break_pos] == ' ':
                return shift, True
    return 0, False

def _add_new_line(txt, break_len=26):
    left_len = len(txt)
    cur_pos = 0
    while left_len > break_len:
        break_pos = cur_pos + break_len
        shift, nodash = _find_break_pos(txt, break_pos)
        if nodash:
            insert_txt = '\n'
        else:
            insert_txt = '-\n'
        new_break_pos = break_pos + shift
        if new_break_pos < len(txt):
            txt = txt[0: new_break_pos] + insert_txt + txt[new_break_pos:]
        cur_pos = new_break_pos + len(insert_txt)
        left_len = len(txt) - cur_pos
    return txt

def plot_dialogue(df, y_start=1.0, max_time=None):
    for row in df.itertuples(index=False, name=None):
        # print(row)
        x1, x2, speaker, value = row
        if speaker == 'INV':
            color = 'r'
        elif speaker == 'PAR':
            color = 'b'
        else:
            color = 'k'
        
        if (max_time is not None) and x1>max_time:
            print("[INFO @ %s]"%__name__, "Exceed max time: %f-%f %s: %s" % (x1, x2, speaker, value))
            continue
        plt.gca().add_line(Line2D([x1, x2],[y_start, y_start], 
                           linewidth=2, marker='+',
                           color=color))
        txt = plt.text((x1+x2)/2.0, y_start + 0.1, _add_new_line(value),
                 fontsize=9, rotation='vertical', va='center', ha='left', wrap=True)  # 
