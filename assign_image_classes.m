function assign_image_classes(image_dir, metadata_dir, label_dir)
%ASSIGN_IMAGE_CLASSES Assign classes 0 (bad) or 1 (good) to images
%   Goes through each of the subfolders in the directories and
%   uses Flickr metadata to assign classes.

bg = 0.4; % Bad to good ratio.
vi = 0.5; % Number of views to Flickr "interestingness"

if (nargin < 3)
   image_dir = ...
       'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\train\images';
   metadata_dir = ...
       'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\train\image_metadata';
   label_dir = ...
       'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\train\labels';
end
fprintf('Reading images from %s\n', image_dir);
fprintf('Reading image metadata from %s\n', metadata_dir);
fprintf('Writing labels to %s\n', label_dir);
fprintf('\n');

% Get categories.
categories = dir(fullfile(image_dir));
num_categories = length(categories) - 2; % ignore '.' and '..' 
fprintf('Found %i subfolders in %s\n', num_categories, image_dir);

% Iterate through and assign classes.
for catInd = 3:(num_categories + 2)
    
    % Get subfolders.
    category = categories(catInd).name;
    category_dir = fullfile(image_dir, category);
    subfolders = dir(category_dir);
    num_subfolders = length(subfolders) - 2;
    
    % Get number of images.
    num_total_images = 1000 * (num_subfolders - 1);
    last_images = dir(fullfile(category_dir, ...
        subfolders(num_subfolders).name, '*.jpg'));
    num_images = num_total_images + length(last_images);
    fprintf('   Found %i images for category %s\n', num_images, category);
    
    % Open metadata file
    metadata_filename = fullfile(metadata_dir, strcat(category, '.txt'));
    fid = fopen(metadata_filename,'r');
    fprintf(' Reading search results in file %s\n', metadata_filename);
    
    ids = cell(1, num_images);
    views = zeros(1, num_images);
    interest = zeros(1, num_images);
    
    % Find image matches, enter metadata.
    count = 0;
    num_processed = 0;
    while num_processed < num_images
        line = fgetl(fid);
        if (~ischar(line))
            break
        end

        %example entry
        % photo: 5431522034 0967acaf0e 5219
        % owner: 96339726@N00
        % title: LifeForms - Solitaire
        % originalsecret: null
        % originalformat: null
        % o_height: null
        % o_width: null
        % datetaken: 2011-02-03 15:42:51
        % dateupload: 1297272645
        % tags: sea sky italy reflection beach nature clouds ...
        % license: 0
        % views: 1109
        % interestingness: 2 out of 557

        if strncmp(line,'photo:',6)
        first_line = line;

        [t,r] = strtok(line);
        [id,r] = strtok(r);
        [secret,r] = strtok(r);
        [server,r] = strtok(r);

        line = fgetl(fid); %owner: line
        second_line = line;
        [t,r] = strtok(line);
        [owner,r] = strtok(r);

        % Get all metadata.
        img_metadata = strvcat(first_line, ... %photo id secret server
                               second_line,... %owner
                               fgetl(fid), ... %title
                               fgetl(fid), ... %original_secret            
                               fgetl(fid), ... %original_format
                               fgetl(fid), ... %o_height
                               fgetl(fid), ... %o_width
                               fgetl(fid), ... %datetaken
                               fgetl(fid), ... %dateupload
                               fgetl(fid), ... %tags
                               fgetl(fid), ... %license
                               fgetl(fid), ... %views
                               fgetl(fid)) ;   %interestingness (string)

        % Find matching image.
        images = [];
        for subInd = 3:(num_subfolders + 2) 
            imgs = dir(fullfile(category_dir, ...
                subfolders(subInd).name, '*.jpg'));
            imgs = {imgs([imgs.isdir] == 0).name};
            images = [images imgs];
        end

        linesplit = strsplit(first_line, ' ');
        photo_id = char(linesplit(2));
        regex = strcat(photo_id, '*');

        result = images(~cellfun(@isempty, ...
            regexpi(images, regex)));

        if size(result,2) == 1
            count = count + 1;
            ids(1, count) = result(1,1);

            views_line = strsplit(char(img_metadata(12,:)), ' ');
            views(1, count) = str2double(views_line(2));

            interest_line = strsplit(char(img_metadata(13,:)), ' ');
            interest(1, count) = str2double(interest_line(2)) / ...
                str2double(interest_line(5));

            save(fullfile(metadata_dir, strcat(category, ...
                '.mat')), 'ids', 'views', 'interest');
            
        end

        num_processed = num_processed + 1;
        fprintf('Processed %i images, %i left, %i saved. \n', ...
            num_processed, num_images - num_processed, count);
    end
 
        
    end
    
    % Truncate.
    ids = ids(1:count);
    views = views(1:find(views, 1, 'last'));
    interest = interest(1:find(interest, 1, 'last'));
    
    % Calculate classes.
    max_views = max(views);
    scores = (vi * (views / max_views)) + ...
        ((1 - vi) * (interest));
    sorted_scores = sort(scores);
    index = floor(length(ids) * bg);
    thres = 0.5 * (sorted_scores(index) + sorted_scores(index + 1));
    labels = scores > thres;
    
    save(fullfile(metadata_dir, strcat(category, ...
        '.mat')), 'ids', 'views', 'interest');
    save(fullfile(label_dir, strcat(category, '.mat')), 'labels');
    fprintf('%i image IDs, metadata, and labels saved.\n', length(ids)); 

    fclose(fid);
end
end

