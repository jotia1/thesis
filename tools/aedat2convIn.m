function [  ] = aedat2convIn( filename, outfile, kx, ky, kz, msps )
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

%% Algorithm
%   load file
%   use aedat2voxel to create dataset
%   Initialise result matrix 
%
%   Iterate through each event
%       grab the n voxels around event and add to input matrix
%       grab the voxels in future of the event and add to label matrix


% Result matrix will be in the form (2 + n, NUM_SAMPLES) as will labels


%% Code
    %   load file
    [allAddr, ts] = loadaerdat(filename);
    [xs, ys, ps] = extractRetina128EventsFromAddr(allAddr);

    % TODO investigate why DVS data is not being monotonically
    xs = xs(200:floor(end/16)) + 1;  % Matlab indexing from 1 not 0
    ys = ys(200:floor(end/16)) + 1;
    ts = ts(200:floor(end/16));
    ps = ps(200:floor(end/16));

    % use aedat2voxel to create dataset
    sizex = 128; sizey = 128;
    aedatData = [xs, ys, ts, ps, [sizex; sizey; zeros(size(xs, 1)-2, 1)]];
    voxelSpatial = 1;
    loaded = aedat2voxel(aedatData, voxelSpatial, voxelSpatial, msps);
    sizex = ceil(sizex / voxelSpatial);  % This shouldn't change in this script
    sizey = ceil(sizey / voxelSpatial);
    data = single(loaded(sizex*2 + 1: sizex*3, :, :));
    clearvars loaded aedatData % Clean up a little


    % Initialise result matrix 
    dataset = [];
    labels = [];
    
    % Init offsets
    xoff = floor(kx / 2);
    yoff = floor(ky / 2);
    last_buck = size(data, 3) - kz;
    
    % Pad array to save indexing
    data = padarray(data, [xoff, yoff]);


    % Iterate through each event
    for i = 1 : numel(xs) - 1
        tbuck = ceil((ts(i) - ts(1)) / (msps*1e3)) + 1;
        x = xs(i); y = ys(i); z = tbuck;
        if z < kz || z > last_buck  % exclude ends 
            continue;
        end
        % grab the n voxels around event and add to input matrix
        xmin = x; xmax = x + kx - 1;
        ymin = y; ymax = y + ky - 1;
        zmin = z - kz + 1; zmax = z + kz;
        datum = data( xmin : xmax, ...
                        ymin : ymax, ...
                        zmin : z );
         label = data( xmin : xmax, ...
                        ymin : ymax, ...
                        z + 1 : zmax );
        dataset = [dataset, reshape(datum, [], 1)];
        % grab the voxels in future of the event and add to label matrix
        labels = [labels, reshape(label, [], 1)];
        
    end
    
    % Save results
    save(outfile, 'dataset', 'labels');
    
end

