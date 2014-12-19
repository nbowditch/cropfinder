function results = cropfinder_test_antioverfit()

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
results_dir = 'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\results\antioverfit';
metadata_dir = 'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\train\image_metadata';
feat_dirs = [{'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\feats\gist'}, {'GIST'}; ...
             {'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\feats\hog'}, {'HOG'}; ...
             {'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\feats\spp'}, {'SPP'}];


categories = dir(fullfile(image_dir));
catInds = [1 3 5 6 15 18 12 21];
temp = categories(2 + catInds);
categories = cell(length(catInds), 1);
for catInd = 1:length(catInds)
    categories{catInd} = temp(catInd).name;
end

num_g = 4;

% Get previous results.
if ~exist(fullfile(results_dir, 'results.mat'), 'file')
    results = zeros(num_g, size(feat_dirs, 1));
    save(fullfile(results_dir, 'results.mat'), 'results');
end
load(fullfile(results_dir, 'results.mat'));

% Get previous weights.
if ~exist(fullfile(results_dir, 'weights.mat'), 'file')
    weights = cell(num_g, size(feat_dirs, 1));
    save(fullfile(results_dir, 'weights.mat'), 'weights');
end
load(fullfile(results_dir, 'weights.mat'));


for gInd = 0 : num_g-1
    fprintf('TESTING FEATURES ON GROUP %i\n', gInd+1);
    
    % Iterate over types of features.
    for featInd = 1:size(feat_dirs, 1)
        fprintf('  FEATURE: %s\n', feat_dirs{featInd, 2});
        if results(gInd + 1, featInd) ~= 0
            fprintf('    RESULTS ALREADY CALCULATED!\n');
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
                feat_type = feat_dirs{featInd, 2};
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
            
            stInd = floor(gInd .* size(feats, 1) ./ num_g) + 1;
            endInd = floor((gInd+1) .* size(feats, 1) ./ num_g);
            trainInds = ones(size(feats, 1), 1);
            trainInds(stInd:endInd) = zeros(endInd-stInd+1, 1);
            xdata = xdata(trainInds == 1, :);
            ydata = ydata(trainInds == 1, :);

            xTrain = [xTrain; xdata];
            yTrain = [yTrain; ydata];
        end
        fprintf('DONE.\n');
        
        %% Logistic regression for training.
        if isempty(weights{gInd+1, featInd})
            fprintf('    DOING LOGISTIC REGRESSION...');
            pred = glmfit(xTrain, yTrain, 'normal');
            pred = (pred + 1) .* 0.5;
            weights{gInd+1, featInd} = pred;
            save(fullfile(results_dir, 'weights.mat'), 'weights');
            fprintf('DONE.\n');
        else
            fprintf('    LOGISTIC REGRESSION RESULTS FOUND.\n');
            pred = weights{gInd+1, featInd};
        end
        
        
        %% Test data.
        fprintf('    GETTING TEST DATA...');
        xTest = [];
        yTest = [];
       for catInd = 1:length(categories)
            category = categories{catInd}; 
        
            if ~exist(fullfile(feat_dir, strcat(category, '.mat')), 'file')
                % Calculate feature.
                feat_type = feat_dirs{featInd, 2};
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
            
            stInd = floor(gInd .* size(feats, 1) ./ num_g) + 1;
            endInd = floor((gInd+1) .* size(feats, 1) ./ num_g);
            testInds = zeros(size(feats, 1), 1);
            testInds(stInd:endInd) = ones(endInd-stInd+1, 1);
            xdata = xdata(testInds == 1, :);
            ydata = ydata(testInds == 1, :);

            xTest = [xTrain; xdata];
            yTest = [yTrain; ydata];
        end
        fprintf('DONE.\n');
        
        %% Test.
        fprintf('    TESTING...');
        xTest = [ones(size(yTest, 1), 1), xTest]; 
        yHat = pred' * xTest';
        mse = sum((yTest - yHat').^2);
        fprintf('DONE. MSE was %d\n', mse);
        
        results(gInd + 1, featInd) = mse;
        save(fullfile(results_dir, 'results.mat'), 'results');
        
    end
    fprintf('\n');
   
end

avgs = sum(results, 1) ./ size(results, 1)
x = [1 2 3 4];
plot(x, results(:,1), x, results(:,2), x, results(:, 3));


end

