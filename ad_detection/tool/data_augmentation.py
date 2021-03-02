#!/usr/bin/env python
# hellolzc 20200408
# Thanks to Eu Jin Lok https://www.kaggle.com/ejlok1/audio-emotion-part-5-data-augmentation
import os, argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

from subprocess import check_call, Popen, CalledProcessError, PIPE
#########################
# Augmentation methods
#########################
def noise(data):
    """
    Adding White Noise.
    """
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.05*np.random.uniform()*np.amax(data)   # more noise reduce the value to 0.5
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data
    
def shift(data):
    """
    Random Shifting.
    """
    s_range = int(np.random.uniform(low=-5, high = 5)*1000)  #default at 500
    return np.roll(data, s_range)
    
def stretch(data, rate=0.8):
    """
    Streching the Sound. Note that this expands the dataset slightly
    If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
    """
    data = librosa.effects.time_stretch(data, rate)
    return data
    
def pitch(data, sample_rate, pitch_change=None):
    """
    Pitch Tuning.
    """
    bins_per_octave = 12
    if pitch_change is None:
        pitch_pm = 2
        pitch_change =  pitch_pm * 2 * (np.random.uniform())
    data = librosa.effects.pitch_shift(data.astype('float64'), 
                                      sample_rate, n_steps=pitch_change, 
                                      bins_per_octave=bins_per_octave)
    return data


RB_CMD = '/home/zhaoci/toolkit/rubberband-1.8.2/bin/rubberband'
def rubberband_stretch_pitch(infile, outfile, stretch=1.0, pitch=1.0):
    """
    See: https://github.com/breakfastquay/rubberband
    For example,
    $ rubberband -t 1.5 -p 2.0 test.wav output.wav

    stretches the file test.wav to 50% longer than its original duration, 
    shifts it up in pitch by one octave, and writes the output to output.wav.

    Several further options are available: run "rubberband -h" for help. In particular, 
    different types of music may benefit from different "crispness" options 
    (-c flag with a numerical argument from 0 to 6).
    """
    check_call([RB_CMD, '-t', str(stretch), '-p', str(pitch), infile, outfile])

def dyn_change(data):
    """
    Random Value Change.
    """
    dyn_change = np.random.uniform(low=-0.5 ,high=7)  # default low = 1.5, high = 3
    return (data * dyn_change)


def speedNpitch_interp(data, speed_fac=None, trim=False):
    """
    Speed and Pitch Tuning.
    """
    if speed_fac is None:
        # you can change low and high here
        length_change = np.random.uniform(low=0.8, high = 1)
        speed_fac = 1.2  / length_change # try changing 1.0 to 2.0 ... =D
    tmp = np.interp(np.arange(0, len(data), speed_fac), np.arange(0,len(data)), data)
    if trim:
        minlen = min(data.shape[0], tmp.shape[0])
        res = np.zeros(data.shape)
        res[0:minlen] = tmp[0:minlen]
    else:
        res = tmp
    return res

'''
2. Extracting the MFCC feature as an image (Matrix format).  
'''
def prepare_data(df, n, aug, mfcc):
    X = np.empty(shape=(df.shape[0], n, 216, 1))
    input_length = sampling_rate * audio_duration
    
    cnt = 0
    for fname in tqdm(df.path):
        file_path = fname
        data, _ = librosa.load(file_path, sr=sampling_rate
                               ,res_type="kaiser_fast"
                               ,duration=2.5
                               ,offset=0.5
                              )

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

        # Augmentation? 
        if aug == 1:
            data = speedNpitch(data)
        
        # which feature?
        if mfcc == 1:
            # MFCC extraction 
            MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
            MFCC = np.expand_dims(MFCC, axis=-1)
            X[cnt,] = MFCC
            
        else:
            # Log-melspectogram
            melspec = librosa.feature.melspectrogram(data, n_mels = n_melspec)   
            logspec = librosa.amplitude_to_db(melspec)
            logspec = np.expand_dims(logspec, axis=-1)
            X[cnt,] = logspec
            
        cnt += 1
    
    return X


def data_aug_main(inpath, outpath, method=0):
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    filenames = os.listdir(inpath)
    filenames = sorted([x for x in filenames if x.endswith('.wav')])
    info_list = []

    for filename in tqdm(filenames):
        file_path = os.path.join(inpath, filename)
        # print(filename, end=' ', flush=True)
        try:
            data, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            print(filename, e)
            continue

        if method == 0:
            rate = 0.9 # + np.random.uniform(low=-0.05, high = 0.05)
            aug_data = speedNpitch_interp(data, rate)  # 0.9是慢放
            new_name = filename[:-4] + '-a01' + '.wav'
            output_fp = os.path.join(outpath, new_name)
            sf.write(output_fp, aug_data, sr, subtype='PCM_16')
            info_list.append((new_name, rate))

            rate = 1.1 # + np.random.uniform(low=-0.05, high = 0.05)
            aug_data = speedNpitch_interp(data, rate)  # 1.1是快放
            new_name = filename[:-4] + '-a02' + '.wav'
            output_fp = os.path.join(outpath, new_name)
            sf.write(output_fp, aug_data, sr, subtype='PCM_16')
            info_list.append((new_name, rate))

            # rate = 0.95  # + np.random.uniform(low=-0.05, high = 0.05)
            # aug_data = speedNpitch_interp(data, rate)  # 0.9是慢放
            # new_name = filename[:-4] + '-a03' + '.wav'
            # output_fp = os.path.join(outpath, new_name)
            # sf.write(output_fp, aug_data, sr, subtype='PCM_16')
            # info_list.append((new_name, rate))

            # rate = 1.05  # + np.random.uniform(low=-0.05, high = 0.05)
            # aug_data = speedNpitch_interp(data, rate)  # 1.1是快放
            # new_name = filename[:-4] + '-a04' + '.wav'
            # output_fp = os.path.join(outpath, new_name)
            # sf.write(output_fp, aug_data, sr, subtype='PCM_16')
            # info_list.append((new_name, rate))
        elif method == 1:
            rate = 0.9 + np.random.uniform(low=-0.05, high = 0.05)
            new_name = filename[:-4] + '-a11' + '.wav'
            output_fp = os.path.join(outpath, new_name)
            rubberband_stretch_pitch(file_path, output_fp, stretch=1.0/rate)

            info_list.append((new_name, rate))

            rate = 1.1 + np.random.uniform(low=-0.05, high = 0.05)
            new_name = filename[:-4] + '-a12' + '.wav'
            output_fp = os.path.join(outpath, new_name)
            rubberband_stretch_pitch(file_path, output_fp, stretch=1.0/rate)

            info_list.append((new_name, rate))
        elif method == 2:
            pitch = 2 + np.random.uniform(low=-0.5, high = 0.5)
            new_name = filename[:-4] + '-a21' + '.wav'
            output_fp = os.path.join(outpath, new_name)
            rubberband_stretch_pitch(file_path, output_fp, pitch=pitch)

            info_list.append((new_name, 1.0))

            pitch = -2 + np.random.uniform(low=-0.5, high = 0.5)
            new_name = filename[:-4] + '-a22' + '.wav'
            output_fp = os.path.join(outpath, new_name)
            rubberband_stretch_pitch(file_path, output_fp, pitch=pitch)

            info_list.append((new_name, 1.0))
    # save info
    file_path = os.path.join(outpath, 'aug_method%d_info.csv' % method)
    df = pd.DataFrame(data=info_list, columns=['filename', 'rate'])
    df.to_csv(file_path, index=False)
    print('All Done!')


if __name__ == '__main__':
    # ./data_augmentation.py -m 0 -i '../../data/high-pass' -o '../../data/wav_hp_aug'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--method',
        type=int,
        default='0',
        help='0: speedNpitch 1: time stretch 2: pitch shift'
    )
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        default='./data/high-pass',
        help='The input directory'
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default='./data/wav_hp_aug',
        help='The output directory'
    )

    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unknown arguments: ', unparsed)

    assert FLAGS.method in [0, 1, 2]
    data_aug_main(FLAGS.input_dir, FLAGS.output_dir, FLAGS.method)
