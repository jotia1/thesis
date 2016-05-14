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
    [ xs, ys, ts, ps, seps ] = trimEvents(filename);

    assert(kx == ky, 'Varying size kernels not supported yet');
    
    events_per_img = 100;
    
%   Init result matrix
    nevents = numel(xs);
    nsamples = floor(nevents / events_per_img);
    inputs = zeros(nsamples, kx * ky, 'single');
    labels = zeros(nsamples, kx * ky, 'single');
    
%   For each pair in sep
    cur_sample = 1;
    for sep = 1:size(seps, 1);
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
        %stride = events_per_img;
       % samples4section = floor((endi - starti - ...
        %    mod(endi, stride) + (stride - mod(starti, stride))) ...
        %    / events_per_img);
        assert(size(ins, 1) == size(labs, 1), 'dataset and labels not conistant');
        inputs(cur_sample : cur_sample + size(ins, 1) - 1, :) = ins;
        labels(cur_sample : cur_sample + size(labs, 1) - 1, :) = labs;
        cur_sample = cur_sample + size(labs, 1);

    end

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
        for i = 500:50:100000
            waitforbuttonpress
            imshow(mat2gray(reshape(train_inputs(i, :), 128, 128)))
        end
    end
    
%   save result matrix   
    timestamp = date;
    save(outfile, 'train_inputs', 'train_labels', ...
                    'test_inputs', 'test_labels', ...
                    'valid_inputs', 'valid_labels', ...
                    'filename', 'kx', 'ky', 'kz', ...
                    'msps', 'k', 'timestamp' ...
                );
    
    
    
    
    