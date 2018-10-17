function H = use_doap(net, patches)
%USE_DOAP Given the DOAP network and a (42xN, 42) matrix of patches,
%compute the descriptor for each patch and return it as an Nx128 matrix.
%   @net: DOAP neural network. A DagNN network.
%   @patches: Array of form 42xN, 42.
%   
%   Outputs:
%   @H: Array of form Nx128. Array of L2 real descriptors.
nbits = 128;        % descriptor dimensions.

% Patches must have the size of 42x42 pixels. A csv file must therefore
% have the size of Nx42x42, where N is the number of patches.
[num_rows, num_cols] = size(patches);
N = single(num_rows / num_cols);

% Reshape the patches to be used with the neural network, then normalize
% them.
P = reshape(patches, 42, 42, 1, N);
P = normalize_patches(P);

% output matrix
H = zeros(nbits, N, 'single');

for t = 1:N
    data = P(:, :, :, t);
    % Evaluate patch.
    net.eval({'input', data});
    
    % Reduce to column form and insert into output matrix H.
    H(:, t) = squeeze(gather(net.vars(end).value));
end

% H is now a 128 x N matrix, meaning each column is one descriptor.
% L2 normalization to get real descriptor
H = bsxfun(@rdivide, H, sqrt(sum(H.^2, 1)));

% Transpose matrix to get one descriptor per row and change the output
% matrix form to N x 128.
H = H.';
end









