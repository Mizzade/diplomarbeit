function setup_matlab_once(dir_compilenn)
%SETUP_MEX_AND_COMPILE Sets the gnu compiler and compiles VLFeat.
%   Needs to be done once at setup phase.
%   Don't call this function manually. It will be called from
%   from the setup script.
mex -setup C++

cd(dir_compilenn);
vl_compilenn
end

