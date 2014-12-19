function crops = cropfinder_sliding_window( im, step, scales)
%CROPFINDER_SLIDING_WINDOW Summary of this function goes here
%   Detailed explanation goes here
    height = size(im, 1);
    width = size(im, 2);
    crop_heights = floor(scales * height);
    crop_widths = floor(scales * width);
    
    numcrops = 0;
    for scalInd = 1:length(scales)
        crop = [0, 0, crop_widths(scalInd), crop_heights(scalInd)];
         while crop(2) + crop_heights(scalInd) < height
            while crop(1) + crop_widths(scalInd) < width
                numcrops = numcrops + 1;
                crop = [crop(1) + step, crop(2), crop_widths(scalInd), crop_heights(scalInd)];
            end
            numcrops = numcrops + 1;
            crop = [0, crop(2) + step, crop_widths(scalInd), crop_heights(scalInd)];
        end
    end
    
    crops = ones(numcrops, 4);
    
    cropInd = 1;
    for scalInd = 1:length(scales)
        crop = [0, 0, crop_widths(scalInd), crop_heights(scalInd)];
        while crop(2) + crop_heights(scalInd) < height
            while crop(1) + crop_widths(scalInd) < width
                crops(cropInd, :) = crop;
                crop = [crop(1) + step, crop(2), crop_widths(scalInd), crop_heights(scalInd)];
                cropInd = cropInd + 1;
            end
            crops(cropInd, :) = crop;
            crop = [0, crop(2) + step, crop_widths(scalInd), crop_heights(scalInd)];
            cropInd = cropInd + 1;
        end
    end
end

