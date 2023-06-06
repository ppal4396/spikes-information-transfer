

mstr = {'Krebs','Waksman','Robbins'};
load(( 'probeLocations.mat'));



tstart = [3811 3633 3323];


mouseID = 1
load(['neuropixel_data/spks/spks',mstr{mouseID},'_Feb18.mat']);

pp = 8; %1:8 probe.
clu = spks(pp).clu; % cluster ids
st  = spks(pp).st;  % spike times in seconds
% height of clusters on the probe
% we will use these to determine the brain area
%The height of cell (from what electrode??)
whp = spks(pp).Wheights(sort(unique(clu)));

%
%This is the ccf region that each electrode is in
ccfRegion = probeLocations(mouseID).probe(pp).ccfOntology;

%this grabs the height of each electrode in VISp5 (V1 layer 5)
wHeight5 =  probeLocations(mouseID).probe(pp).siteCoords(contains(ccfRegion,'VISp5'),2);

if isempty(wHeight5)
    wHeight5 = 0;
end

%Assume anything within this boundary (excl those on the
%boundaries to be safe)
gdCellsID5 = find(whp < max(wHeight5) & whp > min(wHeight5));

%Create empty cell to store spikeTimes
l5Cells = cell(1,numel(gdCellsID5));

for nn = 1:numel(gdCellsID5)
    
    spikeTimes = sort(st(clu == gdCellsID5(nn)));
    
    
    l5Cells{nn} = spikeTimes;
    
    
    neuronid = ['mouse_' num2str(mouseID) '_probe_' num2str(pp) 'layer_5_cell_' num2str(nn)];
    
    writematrix(spikeTimes,neuronid)
    
end

%%

%this grabs the height of each electrode in VISp6
wHeight6 =  probeLocations(mouseID).probe(pp).siteCoords(contains(ccfRegion,'VISp6'),2);

if isempty(wHeight6)
    wHeight6 = 0;
end


%Assume anything within this boundary (excl those on the
%boundaries to be safe)
gdCellsID6 = find(whp < max(wHeight6) & whp > min(wHeight6));

%Create empty cell to store spikeTimes
l6Cells = cell(1,numel(gdCellsID6));

for nn = 1:numel(gdCellsID6)
    
    spikeTimes = sort(st(clu == gdCellsID6(nn)));
    
    
    l6Cells{nn} = spikeTimes;
    
     neuronid = ['mouse_' num2str(mouseID) '_probe_' num2str(pp) 'layer_6_cell_' num2str(nn)];
    
    writematrix(spikeTimes,neuronid)
    
    
end

%%
%this grabs the height of each electrode in VISp2/3
wHeight23 =  probeLocations(mouseID).probe(pp).siteCoords(contains(ccfRegion,'VISp2'),2);

if isempty(wHeight23)
    wHeight23 = 0;
end

%Assume anything within this boundary (excl those on the
%boundaries to be safe)
gdCellsID23 = find(whp < max(wHeight23) & whp > min(wHeight23));

%Create empty cell to store spikeTimes
l23Cells = cell(1,numel(gdCellsID23));

for nn = 1:numel(gdCellsID23)
    
    spikeTimes = sort(st(clu == gdCellsID23(nn)));
    
    
    l23Cells{nn} = spikeTimes;
    
     neuronid = ['mouse_' num2str(mouseID) '_probe_' num2str(pp) 'layer_23_cell_' num2str(nn)];
    
    writematrix(spikeTimes,neuronid)
    
    
end

%%
%this grabs the height of each electrode in VISp4
wHeight4 =  probeLocations(mouseID).probe(pp).siteCoords(contains(ccfRegion,'VISp4'),2);

if isempty(wHeight4)
    wHeight4 = 0;
end

%Assume anything within this boundary (excl those on the
%boundaries to be safe)
gdCellsID4 = find(whp < max(wHeight4) & whp > min(wHeight4));

%Create empty cell to store spikeTimes
l4Cells = cell(1,numel(gdCellsID4));

for nn = 1:numel(gdCellsID4)
    
    spikeTimes = sort(st(clu == gdCellsID4(nn)));
    
    
    l4Cells{nn} = spikeTimes;
    
     neuronid = ['mouse_' num2str(mouseID) '_probe_' num2str(pp) 'layer_4_cell_' num2str(nn)];
    
    writematrix(spikeTimes,neuronid)
    
    
end

