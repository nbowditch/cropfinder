function cropfinder_process()

image_dir = 'C:\Users\Nate\Development\python\cropfinder\flickr_interesting3';
metadata_dir =  'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\train\image_metadata';
query_dir = 'C:\Users\Nate\Development\python\cropfinder\search_results2';
label_dir =  'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\train\labels';
score_dir =  'C:\Users\Nate\Development\cropfinder\datasets\cropfinder\train\scores';

% Pick a random category.
categories = dir(fullfile(image_dir));
num_categories = length(categories) - 2; % ignore '.' and '..' 
fprintf('Found %i subfolders in %s\n', num_categories, image_dir);

% Iterate through and assign classes.
for catInd = 3:(num_categories + 2)
    category = categories(catInd).name;
    if (~exist(fullfile(metadata_dir, strcat(category, '.mat')), 'file'))
        fprintf('assigning classes for category %s\n', category);
        cropfinder_process_image_queries(image_dir, query_dir, ...
            metadata_dir, label_dir, score_dir, category);        
    else
        fprintf('classes already assigned for category %s\n', category);
    end
end

fprintf('DONE!\n');



end

