#!/bin/bash
# Run Matlab script `../tool/peak_normalization.m` before this script.

# sox_bin_dir="/e/NELSLIP/toolkit/sox-14.4.2"

cd ../../ws_en/
project_dir=`pwd`

sil_wav=/dev/shm/sil.wav
tmp_wav=/dev/shm/tmp.wav

sox -n -r 16k -b 16 -c 1 $sil_wav trim 0.0 0.5

mkdir -p ./data/high-pass

cd $project_dir/data/wav_norm/
for file in *.wav; do
    #echo $file
    c=${file}
    echo $c
    # ./sox.exe ../sox/in.wav -r 16k -b 16  ../out.wav  highpass 60 remix -
    # sox $tmp_wav2 $tmp_wav3 norm -3 highpass 65
    sox $c $tmp_wav highpass 65
    sox $tmp_wav $sil_wav $project_dir/data/high-pass/${c%.*}.wav
done


rm $sil_wav
rm $tmp_wav

