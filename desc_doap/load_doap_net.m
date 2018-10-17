function obj = load_doap_net(path_to_net, path_to_layers)
%% LOAD_DOAP_NET This is a custom loading function to load
% a given neural net from the paper "Local descriptors
% optimized for average precision". It returns a 
% dagnn.Daggn net.
% Note: You have to make sure vlfeat's vl_setup and 
% matconvnet's vl_setupnn have been run previously.
% @PARAM path_to_net: Path to the .mat file containing
%   the DOAP-net.
% @PARAM path_to_layers: path to all layer classes used 
%   int that DOAP-net.

addpath(path_to_layers);

netStructure = load(path_to_net);
s = netStructure.net;
s.layers(9).type = 'PaddedBilinear';
s.layers(end) = [];

if isstruct(s)
  assert(isfield(s, 'layers'), 'Invalid model.');
  if ~isstruct(s.layers)
    warning('The model appears to be `simplenn` model. Using `fromSimpleNN` instead.');
    obj = dagnn.DagNN.fromSimpleNN(s);
    return;
  end
  obj = dagnn.DagNN() ;
  for l = 1:numel(s.layers)
    constr = str2func(s.layers(l).type) ;
    block = constr() ;
    block.load(struct(s.layers(l).block)) ;
    obj.addLayer(...
      s.layers(l).name, ...
      block, ...
      s.layers(l).inputs, ...
      s.layers(l).outputs, ...
      s.layers(l).params,...
      'skipRebuild', true) ;
  end
  obj.rebuild();
  if isfield(s, 'params')
    for f = setdiff(fieldnames(s.params)','name')
      f = char(f) ;
      for i = 1:numel(s.params)
        p = obj.getParamIndex(s.params(i).name) ;
        obj.params(p).(f) = s.params(i).(f) ;
      end
    end
  end
  if isfield(s, 'vars')
    for f = setdiff(fieldnames(s.vars)','name')
      f = char(f) ;
      for i = 1:numel(s.vars)
        varName = s.vars(i).name;
        if ~strcmp(varName, 'labels') && ~strcmp(varName, 'objective')
            p = obj.getVarIndex(s.vars(i).name) ;
            obj.vars(p).(f) = s.vars(i).(f) ;
        end
      end
    end
  end
  for f = setdiff(fieldnames(s)', {'vars','params','layers'})
    f = char(f) ;
    obj.(f) = s.(f) ;
  end
elseif isa(s, 'dagnn.DagNN')
  obj = s ;
else
  error('Unknown data type %s for `loadobj`.', class(s));
end

% Set net's mode to `test`.
obj.mode = 'test';

% Get index of the layer with the output variable `logits`
ind = obj.getVarIndex('logits'); % 31
obj.vars(ind).precious = 1;

