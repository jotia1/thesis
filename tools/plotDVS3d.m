function [ ] = plotDVS3d( filename, start, finish, polstr )
%PLOTDVS3D Plot DVS data from start events until end
%   Plot a dvs data file from start events until end events with ploarity
%   equal to pol (being 'pos', 'neg', 'both')

    [allAddr, allTs] = loadaerdat(filename);
    %extracts all the information from the address matrices in the form
    %[xcoordinate, ycoordinate, polarity]
    [infomatrix1, infomatrix2, infomatrix3] = extractRetina128EventsFromAddr(allAddr);
    %times are in us
    allTs = int32(allTs);

    xs = infomatrix1(start:finish,:);
    ys = infomatrix2(start:finish,:);
    ts = allTs(start:finish,:);
    ps = infomatrix3(start:finish,:);

    if strcmp(polstr, 'pos') == 0 
        xs = xs(ps==1,:);
        ys = ys(ps==1,:);
        ts = ts(ps==1,:);
    elseif strcmp(polstr, 'neg') == 0
        xs = xs(ps==-1,:);
        ys = ys(ps==-1,:);
        ts = ts(ps==-1,:);
    elseif ~strcmp(polstr, 'both') == 0
        disp('Polarity setting not recognised, must be pos, neg or both.');
        return;
    end

    
    figure
    plot3(xs, ys, ts,'*', 'MarkerSize',2)
    title(strcat('plot of: ', filename)) 
    xlabel('xs', 'fontsize',14,'fontweight','bold','color',[1 0 0])
    ylabel('ys','fontsize',14,'fontweight','bold','color',[0 0 0]) 
    zlabel('Time','fontsize',14,'fontweight','bold','color',[0 0 1]) 
end

