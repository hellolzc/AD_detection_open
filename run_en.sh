#!/bin/bash
# hellolzc 2018/10/1
project_dir=`pwd`
workspace=${project_dir}/ws_en/

# 1. Copy data
# 1.1 Wave pre-processing
cd ad_detection/prepare_en/
./copy_data.sh
# Use audition to normalize MP3 files and convert them to mono audio. Place them in 'wav_mono' dir.
# Use Matlab script to normalize waves again. Place output files in `wav_mono_norm` dir.
# Matlab script: ad_detection/tool/peak_normalization.m
matlab -nodisplay -r  "cd ../ws_en/data; peak_normalization"

# Do high-pass filtering on the audio. Place output files in `high-pass` dir.
./convert_momo_part2.sh

# LSA语音增强效果没有明显提升, 可以考虑不做
# 语音增强, 增强后文件放 wav_hp_lsa


# 1.2 Convert chat files to TSV files.
cd ${project_dir}/ad_detection/prepare_en/
./CHAT2CSV.py


# 2. Generate list for all data avaliable.
echo 'Generate namelist_auto.txt...'
cd ${workspace}/list/
ls -1 ../data/tsv/ | grep tsv > namelist_auto.txt
LANG=zh_CN.utf-8 sort namelist_auto.txt -o namelist_auto.txt
sed -i "s/.tsv//g" namelist_auto.txt
#  Cheack `namelist_auto.txt` and rename it to `namelist.txt` manually.

# 5. 音频切分
# 使用slice_csv_and_wav.py切分data/high-pass目录下下音频
# mkdir ../data/speech_keep_b
./slice_csv_and_wav.py keep_b
# mkdir ../data/speech_slice_wav
./slice_csv_and_wav.py slice_wav



# 6. Acoustic Feature

# Extracted acoustic features with OpenSMILE: egemaps,IS09,IS10,IS11,IS12,CPE16
cd ../opensmile/scripts
rm ../audio_features_egemaps/*.csv
# Use scripts in `ad_detection/tool/opensmile_scripts`. It may take a long time.
# Refer to `extract_all.sh`


# 7. 时长特征 （可跟上一步同时进行）
cd ${project_dir}/ad_detection/tool
# use praat to calculate speech rate
~/toolkit/praat_nogui --run ./praat_speech_rate.txt -25 2 0.3 no $workspace/data/speech_slice_wav/ > $workspace/data/speechrate.csv
mkdir -p $workspace/data/tsv_sr/
./add_nsyll_to_tsv.py
# 提取时长统计量
./extract_duration_features.py
# code/Interview_Check.ipynb

# 提取时长语速特征

# 8. 文本特征 （可跟上一步同时进行）
# 先分词，再提文本特征
# ./extract_linguistic_feature.py feat

# 提取pos特征类型 cfg特征
# ./extract_linguistic_feature.py seg
# 先tmux打开一个java server端口
# ./pos_tag_and_parsing.py


####################### PART 2 ############################
# 1. 合并特征
cd ${project_dir}/ad_detection/mlcode
python merge_feature.py 'list'
# 记得检查
python merge_feature.py 'all'

# 2. 语言模型 训练模型
# notebook/ML*.ipynb

# 3. LLD特征合并
cd ../dlcode
./merge_frame_feature.py -c feature_selected_config_2.json
# 切割 hdf5 中特征成小段，到新的 hdf5 文件中去
./slice_wav_feature.py -m 3 -s 41 -j 10 -f CPE16_selected3
# 给新的 hdf5 文件生成 label
# ./gen_label_for_aug.py -i <input_df> -f ../../data/acoustic_CPE16_selected3_041_010.hdf5 -o <output>

./merge_frame_feature.py -i ../../opensmile/audio_features_mfcc_aug -o ../../data/acoustic_mfcc_aug.hdf5

# 语音增强，生成新的语音
./data_augmentation.py -m 0 -i '../../ws_en/data/high-pass' -o '../../ws_en/data/wav_hp_aug'
# extract features using openSMILE
cd ${project_dir}/ad_detection/tool/opensmile_scripts


# Save selected features to hdf5 files
cd ${project_dir}/ad_detection/dlcode
./merge_frame_feature.py -c feature_selected_config_3_aug.json



# 4. Run jupyter notebooks
# notebook/*.ipynb

