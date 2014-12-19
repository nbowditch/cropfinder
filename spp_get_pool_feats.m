function feat = spp_get_pool_feats( im, spp_model )
% Adapted from spp code written by Ross Girshick
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Shaoqing Ren
% 
% This file is part of the SPP code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

th = tic();
fast_mode = true;
[width, height] = size(im);
boxes = [0, 0, width, height];

if isempty(boxes)
    dets = {};
    return;
end

% extract features from candidates (one row per candidate box)
th = tic();

% calc_fc_in_matlab = true;
feat = spp_features_convX_anysize(im, [], false);
feat = spp_features_convX_to_poolX_anysize(spp_model.spp_pooler, feat, boxes, false);
feat = feat';
end