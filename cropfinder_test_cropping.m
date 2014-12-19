function results = cropfinder_test_cropping()

% Load our spp model.
spp_model_file = '.\data\spp_model\VOC2007\spp_model.mat';
if ~exist(spp_model_file, 'file')
  error('%s not exist ! \n', spp_model_file);
end
try
    load(spp_model_file);
catch err
    fprintf('load spp_model_file : %s\n', err.message);
end
caffe_net_file     = fullfile(pwd, 'data\cnn_model\Zeiler_conv5\Zeiler_conv5');
caffe_net_def_file = fullfile(pwd, 'data\cnn_model\Zeiler_conv5\Zeiler_spm_scale224_test_conv5.prototxt');

use_gpu = false;
if use_gpu
    clear mex;
    g = gpuDevice(1);
end

caffe('init', caffe_net_def_file, caffe_net_file);
caffe('set_phase_test');
if use_gpu
    spp_model.cnn.layers = spp_layers_in_gpu(spp_model.cnn.layers);
    caffe('set_mode_gpu');
else
    caffe('set_mode_cpu');
end
spp_build();

image_dir = 'C:\Users\Nate\Development\python\cropfinder\flickr_interesting3';
score_dir =  'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\train\scores';
label_dir =  'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\train\labels';
results_dir = 'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\results\cropping';
metadata_dir = 'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\train\image_metadata';
feat_dirs = [{'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\feats\gist'}, {'GIST'}; ...
             {'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\feats\hog'}, {'HOG'}; ...
             {'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\feats\spp'}, {'SPP'}];
test_dir = 'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\test\imgs';
scales = [0.6 0.8 0.7 0.9];
num_scales = length(scales);

% Get categories.
categories = dir(fullfile(image_dir));
catInds = [1 3 5 6 15 18 12 21];
temp = categories(2 + catInds);
categories = cell(length(catInds), 1);
for catInd = 1:length(catInds)
    categories{catInd} = temp(catInd).name;
end

% Get test images.
t = dir(fullfile(test_dir));
test_images = cell(length(t) - 2, 1);
for i = 3:length(t)
    test_images{i-2} = t(i).name;
end


% Get previous results.
if ~exist(fullfile(results_dir, 'results.mat'), 'file')
    results = cell(length(test_images), num_scales, size(feat_dirs, 1));
    save(fullfile(results_dir, 'results.mat'), 'results');
end
load(fullfile(results_dir, 'results.mat'));

% Get previous weights.
if ~exist(fullfile(results_dir, 'weights.mat'), 'file')
    weights = cell(num_scales, size(feat_dirs, 1));
    save(fullfile(results_dir, 'weights.mat'), 'weights');
end
load(fullfile(results_dir, 'weights.mat'));

for imInd = 1:length(test_images)
fprintf('CROPPING IMAGE %s\n', test_images{imInd});
image(imread(fullfile(test_dir, test_images{imInd})));

for scaleInd = 1 : num_scales
    fprintf('  USING SCALE %d\n', scales(scaleInd));
    
    % Iterate over types of features.
    for featInd = 1:size(feat_dirs, 1)
        fprintf('  FEATURE: %s\n', feat_dirs{featInd, 2});
        feat_type = feat_dirs{featInd, 2};
        if ~isempty(results{imInd, scaleInd, featInd})
            fprintf('    BEST CROP ALREADY CALCULATED!\n');
            im = imread(fullfile(test_dir, test_images{imInd}));
            cropped = imcrop(im, results{imInd, scaleInd, featInd});
            imwrite(cropped, fullfile(results_dir, feat_type, strcat(feat_type, ...
                '_', int2str(100*scales(scaleInd)), '_', int2str(imInd), '.jpg')));
            continue;
        end;
        feat_dir = feat_dirs{featInd, 1};
        
        %% Train data.
        fprintf('    GETTING TRAIN DATA...');
        xTrain = [];
        yTrain = [];
        for catInd = 1:length(categories)
            category = categories{catInd};        
            if ~exist(fullfile(feat_dir, strcat(category, '.mat')), 'file')
                % Calculate feature.
                cropfinder_save_feats(image_dir, feat_dir, metadata_dir, ...
                     label_dir, score_dir, category, spp_model, feat_type);
            end

            load(fullfile(feat_dir, strcat(category, '.mat')));
            load(fullfile(score_dir, strcat(category, '.mat')));
           
            xdata = zeros(size(feats, 1), size(feats{1}, 2));
            if size(xdata(1,:)) ~= size(feats{1}, 1)
                xdata = zeros(size(feats, 1), size(feats{1}, 1));
                for i = 1 : size(feats, 1)
                    feats{i} = feats{i}';
                end
            end
            
            for i = 1 : size(feats, 1)
                xdata(i,:) = feats{i};   
            end
            ydata = scores';
            

            xTrain = [xTrain; xdata];
            yTrain = [yTrain; ydata];
        end
        fprintf('DONE.\n');
        
        %% Logistic regression for training.
        if isempty(weights{scaleInd, featInd})
            fprintf('    DOING LOGISTIC REGRESSION...');
            pred = glmfit(xTrain, yTrain, 'normal');
            pred = (pred + 1) .* 0.5;
            weights{scaleInd, featInd} = pred;
            save(fullfile(results_dir, 'weights.mat'), 'weights');
            fprintf('DONE.\n');
        else
            fprintf('    LOGISTIC REGRESSION RESULTS FOUND.\n');
            pred = weights{scaleInd, featInd};
        end
        
        
        %% Crop.
        fprintf('    FINDING BEST CROP...');
        im = imread(fullfile(test_dir, test_images{imInd}));
        windows = cropfinder_sliding_window(im, 10, scales(scaleInd));
        fprintf('IN %i CROPS...', length(windows));
        scores = zeros(length(windows), 1);

        for i = 1:length(windows)
            scores(i) = cropfinder_get_score(imcrop(im, windows(i, :)), ...
                pred, feat_type, spp_model);
        end

        [~, maxInd] = max(scores);
        results{imInd, scaleInd, featInd} = windows(maxInd, :);
        save(fullfile(results_dir, 'results.mat'), 'results');
        fprintf('DONE.\n');
        
        cropped = imcrop(im, results{imInd, scaleInd, featInd});
        imwrite(cropped, fullfile(results_dir,  feat_type, strcat(feat_type, ...
            '_', int2str(100*scales(scaleInd)), '_', int2str(imInd), '.jpg')));

        
    end
    fprintf('\n');
   
end

end



end

