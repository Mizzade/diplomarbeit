function use_doap_with_csv(vlfeat_root, matconvnet_root, path_to_net, path_to_layers, input_dir, output_dir)
%USE_DOAP_WITH_CSV Given the input folder, find all csv-files in the
%subfolders, load the patches, use the DAOP Descriptor and save the
%resulting descriptor files as csv in the output directory.
%   @vlfeat_root: Absolute path to VLFeat folder.
%   @matconvnet_root: Absolute path to MatConvNet folder.
%   @path_to_net: Absolute path to DOAP .mat file.
%   @path_to_layers: Absolute path to the folder containing custom DOAP
%   Layers.
%   @input_dir: Absolute path to main folder, containing subfolders with
%   csv files. Each csv file should be the patches of an image and have the
%   form of 42xN rows and 42 columns.
%   @output_dir: Absolute path of the output folder.
% Setup VLFeat and MatConvNet.

% VLFeat must already be installed otherwise this
% line will fail.
run(fullfile(strcat(vlfeat_root, '/toolbox/vl_setup')));
run(fullfile(strcat(matconvnet_root, '/matlab/vl_setupnn.m')));

net = load_doap_net(path_to_net, path_to_layers);

% filelist = dir(fullfile(input_dir, '*.csv'))
dirlist = dir(input_dir);

% Given the input folder, find all folders and in those folders
% use the csv files (if any) to compute the doap descriptors.
% Save the output csv files inside the output folder with the
% same structure. The output csv file's name is the same but
% has an '_doap' added.
for i = 1:length(dirlist)
    dirName = dirlist(i).name;
    if (~strcmp(dirName, '.') && ~strcmp(dirName, '..')) > 0
        if dirlist(i).isdir
           % Get a list of csv files in that directory.
           filelist = dir(fullfile(strcat(input_dir, '/', dirName), '*.csv'));
           if ~isempty(filelist)
              % Create a corresponding folder in the output folder if the
              % files exist and the output folder not already exists.
              if ~exist(strcat(output_dir, '/', dirName), 'dir')
                mkdir(strcat(output_dir, '/', dirName));
              end

              % Iterate over all files in the current folder.
              for j = 1:length(filelist)
                 file = filelist(j);

                 % Load csv file
                 fileName = file.name;

                 folderName = file.folder;
                 patches = single(csvread(strcat(folderName, '/', fileName)));

                 % Compute the descriptors for each patch.
                 H = use_doap(net, patches);

                 % Get name of the file without extension.
                 [~, n, ~] = fileparts(fileName);

                 % Create new name for output file.
                 outFileName = strcat(n, '_doap.csv');

                 % Write the resulting matrix back as csv file.
                 csvwrite(strcat(output_dir, '/', dirName, '/', outFileName), H);
              end
           end
        end
    end
end


