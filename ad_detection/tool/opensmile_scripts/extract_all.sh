#!/bin/bash

# # 提取eGeMAPs特征统计量
# ./extract_is09etc_features.py egemaps -i ../../data/high-pass/
# ./extract_is09etc_features.py egemaps -i ../../data/wav_hp_lsa/ -n lsa
# ./convert_stat_format.py  '../func_egemaps_.csv'  '../../fusion/acoustic_egemaps.csv'
# ./convert_stat_format.py  '../func_egemaps_lsa.csv'  '../../fusion/acoustic_egemaps_lsa.csv'


# # 提取其他音频特征

# 消耗时间太长，建议分多个shell执行
# 提取单个被试的
:<<BLOCK
./extract_is09etc_features.py CPE16   -i ../../data/speech_keep_b/ -n keep_b # IS13
./extract_is09etc_features.py egemaps -i ../../data/speech_keep_b/ -n keep_b

./convert_stat_format.py    '../func_CPE16_keep_b.csv'    '../../fusion/acoustic_CPE16_keep_b.csv'
./convert_stat_format.py  '../func_egemaps_keep_b.csv'  '../../fusion/acoustic_egemaps_keep_b.csv'

./extract_is09etc_features.py CPE16   -i ../../data/speech_extract_b/ -n extract_b # IS13
./convert_stat_format.py  '../func_CPE16_extract_b.csv'  '../../fusion/acoustic_CPE16_extract_b.csv'
BLOCK

# 主试和被试混合
rundir=`pwd`
workspace=$(cd ../../../ws_en/;pwd)
# Original wavs
./extract_is09etc_features.py CPE16   -i $workspace/data/high-pass/ -o $workspace/opensmile/  # IS13
./convert_stat_format.py  $workspace/opensmile/func_CPE16_.csv  $workspace/fusion/acoustic_CPE16_.csv

# Augmentated wavs
./extract_is09etc_features.py CPE16   -i $workspace/data/wav_hp_aug/ -o $workspace/opensmile/ -n aug # IS13
# copy original features to augmentated features directory
cd $workspace
cat ./opensmile/func_CPE16_.csv | sed '1d' >> ./opensmile/func_CPE16_aug.csv
cd $rundir
./convert_stat_format.py  $workspace/opensmile/func_CPE16_aug.csv  $workspace/fusion/acoustic_CPE16_aug.csv


