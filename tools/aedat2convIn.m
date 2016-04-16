function [  ] = aedat2convIn( filename, outfile, kx, ky, kz, msps, k )
%AEDAT2CONVIN Given and aedat file label each event as <x,y, p0, ..., pn>
%   where n represents the number of voxels per sample. 
%   filename is the aedat file to be opened.
%   kx, ky, kz represent the size of the box around the event to take
%   msps is the milliseconds to slice the video into
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
    
%   Init result matrix
    nevents = numel(xs);
    inputs = zeros(nevents, kx * ky);
    labels = zeros(nevents, kx * ky);
    
%   For each pair in sep
    for sep = 1:size(seps, 1);
        starti = seps(sep, 1);
        endi = seps(sep, 2);
        kernel_size = kx; % Not currently supporting unevent kernels
%       expDecay()  % get decayed sections
        [ ins, labs ] = expDecay(starti, endi, xs, ys, ts, kernel_size, ...
                                msps, k);
%       add to result matrix
        inputs(starti : endi, : , :) = reshape(ins, [], kx * ky);
        labels(starti : endi, :, :) = reshape(labs, [], kx * ky);

    end
%
%   save result matrix   
    timestamp = date;
    save(outfile, 'inputs', 'labels', 'filename', 'kx', ...
                    'ky', 'kz', 'msps', 'k', 'timestamp');
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
%% Algorithm
%   load file
%   use aedat2voxel to create dataset
%   Initialise result matrix 
%
%   Iterate through each event
%       grab the n voxels around event and add to input matrix
%       grab the voxels in future of the event and add to label matrix


% Result matrix will be in the form (2 + n, NUM_SAMPLES) as will labels    
%     
%     
%     % Handled in trimEvents now...
% %     % TODO investigate why DVS data is not being monotonically
% %     xs = xs(200:floor(end)) + 1;  % Matlab indexing from 1 not 0
% %     ys = ys(200:floor(end)) + 1;
% %     ts = ts(200:floor(end));
% %     ps = ps(200:floor(end));
% 
%     % use aedat2voxel to create dataset
%     sizex = 128; sizey = 128;
%     aedatData = [xs, ys, ts, ps, [sizex; sizey; zeros(size(xs, 1)-2, 1)]];
%     voxelSpatial = 1;
%     loaded = aedat2voxel(aedatData, voxelSpatial, voxelSpatial, msps);
%     sizex = ceil(sizex / voxelSpatial);  % This shouldn't change in this script
%     sizey = ceil(sizey / voxelSpatial);
%     data = single(loaded(sizex*2 + 1: sizex*3, :, :));
%     clearvars loaded aedatData % Clean up a little
% 
% 
%     % Initialise result matrix 
%     dataset = [];
%     labels = [];
%     
%     % Init offsets
%     xoff = floor(kx / 2);
%     yoff = floor(ky / 2);
%     last_buck = size(data, 3) - kz;
%     
%     % Pad array to save indexing
%     data = padarray(data, [xoff, yoff]);
%     dataset = zeros(numel(xs), kx * ky * kz);
%     labels = zeros(numel(xs), kx * ky * kz);
% 
%     % Iterate through each event
%     start = 1;
%     for i = 1 : numel(xs) - 1
%         tbuck = ceil((ts(i) - ts(1)) / (msps*1e3)) + 1;
%         x = xs(i); y = ys(i); z = tbuck;
%         if z < kz
%             start = i;
%             continue;
%         end
%         if z > last_buck  % exclude ends 
%             break;
%         end
%         endd = i;
%         % grab the n voxels around event and add to input matrix
%         xmin = x; xmax = x + kx - 1;
%         ymin = y; ymax = y + ky - 1;
%         zmin = z - kz + 1; zmax = z + kz;
%         datum = data( xmin : xmax, ...
%                         ymin : ymax, ...
%                         zmin : z );
%                   
%         % Apply decay to datum (over 30ms so can do same as other funct)
%         
%         label = data( xmin : xmax, ...
%                         ymin : ymax, ...
%                         z + 1 : zmax );
%                     
%         dataset(i, :) = reshape(datum, 1, []);
%         labels(i, :) = reshape(label, 1, []);
%         
%     end
%     dataset = dataset(start : endd, :);
%     labels = labels(start : endd, :);
%     
%     % Save results
%     save(outfile, 'dataset', 'labels');
%     
% end
% 
