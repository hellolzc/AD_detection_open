#!/bin/bash
# sox_bin_dir="/e/NELSLIP/toolkit/sox-14.4.2"
cd ../../ws_en/
project_dir=`pwd`
mkdir -p ./data/wav_mp3

# tmp_wav=/dev/shm/tmp.wav

datadir[0]=$project_dir/data_ori/ctrl/cookie
datadir[1]=$project_dir/data_ori/dementia/cookie
for loop in 0 1
do
    cd ${datadir[$loop]}
    for file in *.mp3; do
        #echo $file
        c=${file}
        echo $c
        # ./sox.exe ../sox/in.wav -r 16k -b 16  ../out.wav  highpass 60 remix -
        # ffmpeg -y -i $c -ac 1 -f flac $project_dir/data/wav_mono_flac/${c%.*}.flac -v 24
        # sox $tmp_wav -r 16k -b 16 $project_dir/data/wav/${c%.*}.wav
        cp $c $project_dir/data/wav_mp3/${c%.*}.mp3
    done
done

# rm $tmp_wav

