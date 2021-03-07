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



# 1.2 Convert chat files to TSV files.
cd ${project_dir}/ad_detection/prepare_en/
./CHAT2CSV.py
# The group of '358-0c' is 'Probable'. Maybe it is a typo.


# 2. Generate list for all data avaliable.
echo 'Generate namelist_auto.txt...'
cd ${workspace}/list/
ls -1 ../data/tsv/ | grep tsv > namelist_auto.txt
LANG=zh_CN.utf-8 sort namelist_auto.txt -o namelist_auto.txt
sed -i "s/.tsv//g" namelist_auto.txt
#  Cheack `namelist_auto.txt` and rename it to `namelist.txt` manually.




# 6. Acoustic Feature

# Extracted acoustic features with OpenSMILE: egemaps,IS09,IS10,IS11,IS12,CPE16
cd ${project_dir}/ad_detection/tool/opensmile_scripts
rm ${workspace}/opensmile/audio_features_egemaps/*.csv
# Use scripts in `ad_detection/tool/opensmile_scripts`. It may take a long time.
# Refer to `extract_all.sh`




####################### PART 2 ############################

# Save selected features to hdf5 files
cd ${project_dir}/ad_detection/dlcode
./merge_frame_feature.py -c feature_selected_config_3_aug.json



# 4. Run jupyter notebooks
# notebook/*.ipynb

