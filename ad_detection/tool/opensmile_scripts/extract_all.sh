#!/bin/bash

# # 提取eGeMAPs特征统计量
# ./extract_is09etc_features.py egemaps -i ../../data/high-pass/
# ./convert_stat_format.py  '../func_egemaps_.csv'  '../../fusion/acoustic_egemaps.csv'


# # 提取其他音频特征



# 主试和被试混合
rundir=`pwd`
workspace=$(cd ../../../ws_en/;pwd)
# Original wavs
./extract_is09etc_features.py CPE16   -i $workspace/data/high-pass/ -o $workspace/opensmile/  # IS13
./convert_stat_format.py  $workspace/opensmile/func_CPE16_.csv  $workspace/fusion/acoustic_CPE16.csv

# Augmentated wavs
./extract_is09etc_features.py CPE16   -i $workspace/data/wav_hp_aug/ -o $workspace/opensmile/ -n aug # IS13
# copy original features to augmentated features directory
cd $workspace
cat ./opensmile/func_CPE16_.csv | sed '1d' >> ./opensmile/func_CPE16_aug.csv
cd $rundir
./convert_stat_format.py  $workspace/opensmile/func_CPE16_aug.csv  $workspace/fusion/acoustic_CPE16_aug.csv


