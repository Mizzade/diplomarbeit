% Wrapper for PaddedBilinear
% (c) 2017 Kun He

classdef PaddedBilinear < dagnn.Layer
    methods
        function outputs = forward(obj, inputs, params)
            % reduce boundary effect by padding
            % 1. pad input images
            sz = size(inputs{1});  % should be SxSx1xN
            % 2. properly modify affine transformation parameters
            %    initial grid now occupies 1/3 the scale in padded image
            if 0
                X  = padarray(inputs{1}, round(sz(1:2)), 'replicate');
                A  = inputs{2} / 3;
            else
                X  = padarray(inputs{1}, round(sz(1:2)/2), 'replicate');
                A  = inputs{2} / 2;
            end
            % 3. send to bilinear sampler
            outputs = vl_nnbilinearsampler(X, A);
            outputs = {outputs};
        end

        function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
            [dX,dG] = vl_nnbilinearsampler(inputs{1}, inputs{2}, derOutputs{1});
            derInputs = {dX,dG};
            derParams = {};
        end

        function outputSizes = getOutputSizes(obj, inputSizes)
            xSize = inputSizes{1};
            gSize = inputSizes{2};
            outputSizes = {[gSize(2), gSize(3), xSize(3), xSize(4)]};
        end

        function obj = PaddedBilinear(varargin)
            obj.load(varargin);
        end
    end
end
