function cropfinder_calc_GISTs( image_dir, feat_dir, metadata_dir, ...
    label_dir, score_dir, category)
%CROPFINDER_CALC_GISTS Summary of this function goes here
%   Detailed explanation goes here

lock = 'lock';
save(fullfile(feat_dir, strcat(category, ...
    '.mat')), 'lock');

category_dir = fullfile(image_dir, category);
subfolders = dir(category_dir);
num_subfolders = length(subfolders) - 2;

load(fullfile(metadata_dir, strcat(category, '.mat')));
if ~exist('ids', 'var')
    fprintf('ERROR, ids not found for category %s\n', category);
    return;
end

% Calculate number of images.
feats = cell(size(ids, 2), 1);

limit = length(ids);
if length(ids) > 2000
    limit = 2000;
end

% GIST params.
clear param
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

for idInd = 1:limit
    filename = ids(idInd);
    found = 0;
    for subInd = 3:(num_subfolders + 2)
        fullname = fullfile(image_dir, category, ...
            subfolders(subInd).name, filename);
        if exist(fullname{1}, 'file')
            %fprintf('   found file!');
            im = imread(fullname{1});
            h = size(im, 1);
            w = size(im, 2);
            if (abs(w - h) < 5)
                im = imresize(im, [256 256]);
            elseif w > h
                im = imresize(im, [192 256]);
            else
                im = imresize(im, [192 256]);
            end         
            
            feats{idInd} = LMgist(im, '', param);
            %fprintf('      GIST %i of %i, size is %i by %i\n', idInd, ...
             %   length(ids), size(feats{idInd}, 1), size(feats{idInd}, 2));
            found = 1;
        end
        if found
            break;
        end
    end
    if ~found
        %fprintf('Image %i not found: %s. Removing metadata, label, and score', idInd, ids{idInd});
        load(fullfile(label_dir, strcat(category, '.mat')));
        load(fullfile(score_dir, strcat(category, '.mat')));
        
        limit = limit - 1;
        feats(idInd) = [];
        ids(idInd) = [];
        interest(idInd) = [];
        views(idInd) = [];
        labels(idInd) = [];
        scores(idInd) = [];
        
        save(fullfile(metadata_dir, strcat(category, ...
            '.mat')), 'ids', 'views', 'interest');
        save(fullfile(label_dir, strcat(category, '.mat')), 'labels');
        save(fullfile(score_dir, strcat(category, '.mat')), 'scores');
    end
end

save(fullfile(feat_dir, strcat(category, '.mat')), 'feats');
%fprintf('saved features for category %s\n', category);

end

