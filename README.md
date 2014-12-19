cropfinder
==========

An automatic image cropper built on the SPP-net convolutional neural network.

To use, you will need to do two things:

1. Create a dataset by querying and downloading Flickr images. I used the code provided by James Hays here:
http://graphics.cs.cmu.edu/projects/im2gps/flickr_code.html
Note that I used a different list of categories more related to photography for the queries. Be sure to keep the metadata
files from the query results, as they are parsed for metadata used in scoring each image. 

2. Clone into the the SPP-net code, found here, following the instructions provided on the page:
https://github.com/ShaoqingRen/SPP_net
The code from this repository should just be moved into the main SPP-net directory.

If you have any questions, feel free to contact me at nathaniel_bowditch@brown.edu.
