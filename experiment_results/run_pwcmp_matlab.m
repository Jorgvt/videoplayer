% run_pwcmp_matlab.m
% Automatically generated script to scale GAIM240 pairwise comparison data using pwcmp.
% Requires pwcmp from https://github.com/mantiuk/pwcmp

clear; clc;

csv_file = 'pwcmp_data.csv';
fprintf('Loading pairwise comparison results from %s...\n', csv_file);
T = readtable(csv_file);

disp('Dataset Overview:');
summary(T);

% If pwcmp toolbox is on the path, perform scaling per scene:
if exist('pw_scale', 'file') || exist('pw_scale_bml', 'file')
    scenes = unique(T.scene);
    for s = 1:length(scenes)
        cur_scene = scenes{s};
        fprintf('\n=== Scaling Scene: %s ===\n', cur_scene);
        T_scene = T(strcmp(T.scene, cur_scene), :);
        
        % Run pwcmp scaling (Thurstone Case V / BML / B-T)
        try
            [scale_jod, stats] = pw_scale(T_scene);
            disp(stats);
        catch ME
            fprintf('Error during scaling: %s\n', ME.message);
        end
    end
else
    fprintf('\nNOTE: pwcmp toolbox is not detected on your MATLAB path.\n');
    fprintf('Please clone https://github.com/mantiuk/pwcmp and add it to your MATLAB path.\n');
end
