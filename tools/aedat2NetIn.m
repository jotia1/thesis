function [  ] = aedat2NetIn( filename, outfile, kx, ky, kz, msps, attentional, decaytype, k )
%AEDAT2CONVIN Given and aedat file label each event as <x,y, p0, ..., pn>
%   where n represents the number of voxels per sample. 
%   filename is the aedat file to be opened.
%   kx, ky, kz represent the size of the box around the event to take
%   msps is the milliseconds to slice the video into
%   decaytype being a string of the decay function to use (linear or exp)
%   attentional is whether this should CENTER around a distinguished event
%   or just use the FULL image. (options true or false)
%
%   EXAMPLE USAGE:
%       aedat2NetIn('../data/gt/dvs_gt5.aedat', 'trylin', 128, 128, 1, 30, false, 'linear', 900000)
%
% Note: kz refers to the depth from the event into the past (and future)
% not the shape of the box around e (which including the future will be 
% 2 * kz deep).
% NOTE: Col 6 is the one with the event...

%% ALGORITHM
%   
%   Init result matrix and load data
%   For each pair in sep
%       expDecay()  % get decayed sections
%       add to result matrix
%
%   save result matrix


%% Code
    %   load file
    [ xs, ys, ts, ~, seps ] = trimEvents(filename);
    fprintf('%d seps found.\n', size(seps, 1));
    
    assert(kx == ky, 'Varying size kernels not supported yet');
    
    events_per_img = 150;
    
%   Init result matrix
    nevents = numel(xs);
    nsamples = floor(nevents / events_per_img);
    inputs = zeros(nsamples, kx * ky, 'single');
    labels = zeros(nsamples, kx * ky, 'single');
    
    num_angles = 8;
    trials_per_angle = 150;
    
%   For each pair in sep
    cur_sample = 1;
    fprintf('Starting Samples');
    trial_starts = zeros(num_angles, 1);
    ti = 1;
    for sep = 1:size(seps, 1);
        if mod(sep, trials_per_angle) == 1;
            trial_starts(ti) = cur_sample;
            ti = ti + 1;
            disp(sep);
        end
        
        if mod(sep, 10) == 0  % Give some feedback on progress
            fprintf('.');
        end
        starti = seps(sep, 1);
        endi = seps(sep, 2);
        
        kernel_size = kx; % Not currently supporting uneven kernels
        
        if attentional;
            [ ins, labs ] = applyConvDecay(starti, endi, xs, ys, ts, ...
                    kernel_size, msps, events_per_img, decaytype, k);
        else
            [ ins, labs ] = applyFullDecay(starti, endi, xs, ys, ts, ...
                    kernel_size, msps, events_per_img, decaytype, k);        
        end
            
%       add to result matrix 
        assert(size(ins, 1) == size(labs, 1), 'dataset and labels not conistant');
        inputs(cur_sample : cur_sample + size(ins, 1) - 1, :) = ins;
        labels(cur_sample : cur_sample + size(labs, 1) - 1, :) = labs;
        cur_sample = cur_sample + size(labs, 1);
        
    end
    
    
    
    gen_analytic = 1;
    if gen_analytic
        num_kernels = num_angles + 1;
        kernels = zeros(kx, kx, 1, num_kernels);

        for i = 1 : num_angles;
            angle_start = trial_starts(i);
            if i == num_angles  % at last angle (section)
                akern = reshape(sum(inputs(angle_start : end, :)), kx, kx);        
            else
                angle_end = trial_starts(i + 1);
                akern = reshape(sum(inputs(angle_start : angle_end, :)), kx, kx);
            end
            akern(6, 6) = 0;
            akern = akern / max(akern(:));
            akern (6, 6) = 1;
            kernels(:, :, 1, i) =  akern;
            imagesc(akern);
        end
        
        % Add noise detector
        noise = zeros(kx, kx);
        noise(6, 6) = 1;
        kernels(:, :, 1, num_kernels) = noise;
        
        % DO SAVE
        fprintf('\nSaving %s\n', [outfile, '_kernels']);
        timestamp = date;
        save([outfile, '_kernels'], 'kernels', 'kx', 'num_kernels', 'timestamp', ...
                    '-v7.3');
        fprintf('\nFinished saving %s\n', outfile);
    end
    
    
    fprintf('\nSegmenting data %s\n', outfile);
%   Shuffle data
    perm = randperm(nsamples);
    inputs = inputs(perm, :);
    labels = labels(perm, :);
    num_train = floor(size(inputs, 1) * 0.7);
    num_valid = floor(size(inputs, 1) * 0.9);
    train_inputs = inputs(1:num_train, :);
    train_labels = labels(1:num_train, :);
    valid_inputs = inputs(num_train + 1 : num_valid, :);
    valid_labels = labels(num_train + 1 : num_valid, :);
    test_inputs = inputs(num_valid + 1 : end, :);
    test_labels = labels(num_valid + 1 : end, :);
    
    
    debug = 0;
    if debug
        disp('disp')
        for i = 2:1:100000
            waitforbuttonpress
            kx = 11;
            imshow(mat2gray([reshape(train_inputs(i, :), kx, kx), ones(kx, 5), reshape(train_labels(i, :), kx, kx)]))
        end
    end
    
%   save result matrix   
    fprintf('\nSaving %s\n', outfile);
    timestamp = date;
    save(outfile, 'train_inputs', 'train_labels', ...
                    'test_inputs', 'test_labels', ...
                    'valid_inputs', 'valid_labels', ...
                    'filename', 'kx', 'ky', 'kz', ...
                    'msps', 'k', 'timestamp', ...
                    '-v7.3');
    fprintf('\nFinished saving %s\n', outfile);
    
    
    
    