function P = normalize_patches(P)
%NORMALIZE_PATCHES Normalizes input patches P.
%   Input are N 42x42 patches in the form of a matrix 
%   with dimension W x H x 1 x N, where W = H = 42.
%   Output is the normalized form of that matrix such
%   that each patch is normalized.
for i = 1:size(P, 4)
    Pi = P(:, :, :, i);
    P(:, :, :, i) = (Pi - mean(Pi(:))) ./ std(Pi(:));
end

