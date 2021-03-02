% WAV NORMSLIZATION TOOL
% hellolzc 20190919
% matlab -nodisplay -r  "choose=1; peak_normalization"
% choose = 1(peak normalization) /2(robust peak normalization) /3(check peak value)

% clc;
clear;
close all;

choose = 2;
in_data_dir = '../../data/wav_mono/';
out_data_dir = '../../data/wav_norm/';

fprintf('\nWAV NORMSLIZATION TOOL v1.0\n');
fprintf('InDir: %s\nOutDir: %s\n', in_data_dir, out_data_dir);

mkdir(out_data_dir);
% get message id by warning('query', 'last') after the warning coming out
warning('off', 'MATLAB:audiovideo:audiowrite:dataClipped');


wave_files = dir([in_data_dir, '*.wav']);
for ith = 1:numel(wave_files)
    fname = wave_files(ith).name;
    wav_file = [in_data_dir,'/',fname];

    [data,fs]=audioread(wav_file);

    % NORMSLIZATION METHOD 1
    % peak normalization
    peak_value =  max(abs(data(:)));
    scale_1 = 1.0 / peak_value;
    if choose == 1
        data_proc = data.*scale_1;
    end

    % RMS normalization
    % data_proc = data./sqrt(mean(data_proc.^2));

    % NORMSLIZATION METHOD 2
    % In order to eliminate the effects of rare and sharp noise,
    % instead of using the peak volume, use the top 0.1% amplitude peak as the reference.
    linear_sorted_data = sort(data.^2, 'descend');
    data_len = length(linear_sorted_data);
    base_point_index = floor(data_len*0.001);  % TODO: average more points
    base_point_value_abs = sqrt(linear_sorted_data(base_point_index));

    % Calculate the logarithm, then draw a histogram
    % log_sorted_data = log(linear_sorted_data);
    % figure(); histogram(log_sorted_data, -20:0.2:0);
    % title(sprintf('histogram of amplitude (log) %s\nOrigin peak value %f, reference point %d', fname, peak_value, base_point_index))
    % hold on; plot(log_sorted_data(base_point_index), 0, 'x');

    % Normalize to -6dB (50% full scale)
    scale_2 = 1.0 / (base_point_value_abs) * 0.5;
    if choose == 2
        data_proc = data.*scale_2;
        if (scale_2 > 3) || (scale_2 <0.7)
            fprintf('Info: %s scale %f\n', fname, scale_2);
        end
    end

    % 1 <= scale_1 <= scale_2
    scale_2_to_1 = scale_2 / scale_1;
    if (scale_2_to_1) >= 4
        fprintf('Warning: %s, scale_1 %f, scale_2 %f, scale_2_to_1 %f\n', fname, scale_1, scale_2, scale_2_to_1);
    end

    if (choose == 1) || (choose == 2)
        audiowrite([out_data_dir, fname], data_proc, fs);
    end

end



