function score = cropfinder_get_score( im, pred, feat_type, spp_model )
%CROPFINDER_GET_SCORE Summary of this function goes here
%   Detailed explanation goes here

score = 0;

%% HOG
if strcmp(feat_type,'HOG') == 1  
    h = size(im, 1);
    w = size(im, 2);
    if (abs(w - h) < 5)
        im = imresize(im, [256 256]);
    elseif w > h
        im = imresize(im, [192 256]);
    else
        im = imresize(im, [192 256]);
    end
    h = size(im, 1);
    w = size(im, 2);
    cellH = h / 32;
    cellW = w / 32;


    feats = extractHOGFeatures(im, 'CellSize', ...
        [cellH cellW], 'BlockSize', [2 2], 'BlockOverlap', ...
        [0 0], 'NumBins', 4);

%% GIST
elseif strcmp(feat_type, 'GIST') == 1
    clear param
    param.orientationsPerScale = [8 8 8 8];
    param.numberBlocks = 4;
    param.fc_prefilt = 4;
    
    h = size(im, 1);
    w = size(im, 2);
    if (abs(w - h) < 5)
        im = imresize(im, [256 256]);
    elseif w > h
        im = imresize(im, [192 256]);
    else
        im = imresize(im, [192 256]);
    end         

    feats = LMgist(im, '', param);
 
%% SPP
elseif strcmp(feat_type, 'SPP') == 1
    feats = spp_get_pool_feats(imresize(im, 0.25), spp_model);

end

if size(feats, 1) > 1
    feats = [1, feats'];
else 
    feats = [1, feats];
end

score = feats * pred;

end

