function use_doap_with_file(vlfeat_root, matconvnet_root, path_to_net, path_to_layers, file_in, file_out)
%USE_DOAP_WITH_FILE Given a file path `file_in` to a csv containing the 
%patches of an image compute the DOAP descriptors for those patches and 
%save them on in `file_out`.
%   @vlfeat_root: Absolute path to VLFeat folder.
%   @matconvnet_root: Absolute path to MatConvNet folder.
%   @path_to_net: Absolute path to DOAP .mat file.
%   @path_to_layers: Absolute path to the folder containing custom DOAP
%   Layers.
%   file_in: Absolute path to csv file containing patches
%   file_out: Absolute path to csv file containing the computed
%   descriptors.

% VLFeat must already be installed otherwise this
% line will fail.
run(fullfile(strcat(vlfeat_root, '/toolbox/vl_setup')));
run(fullfile(strcat(matconvnet_root, '/matlab/vl_setupnn.m')));

% Load DOAP network
net = load_doap_net(path_to_net, path_to_layers);

% Load csv file
patches = single(csvread(file_in));

% Compute the descriptors for the patches
H = use_doap(net, patches);

%Write resulting matrix back as csv file.
csvwrite(file_out, H);