## Daily Notes (a three-week journal from start of project to completion)

* W1D1 *
    - introductory project discussion between SIT Academy and SouthPole
    - exploration of Google Earth Engine (GEE) interface, documentation, and datasets 
* W1D2 *
    - familiarization with the GEE Python API
    - writing scripts for data visualization and export 
    - overnight export of same-day Landsat-Sentinel images from Zurich and Amazon regions (68*2 + 10*2 images in 2.5 hours)
* W1D3 *
    - compilation of questions for SouthPole
    - exploration of other data visualization libraries (gdal and earthpy)
    - exploration of images to achieve similar contrast 
* W1D4 *
    - working on selecting non-cloudy images automatically
    - finding forested yet low-cloud areas across the globe, downloading image pairs
    - reading about scaling/luminance adjustments
* W1D5 *
    - we made a notebook called tif_exporter_moving square. The notebook does the following: 
        - export multiple images from landsat and sentinel with a given LAT/LONG coordinate,
        - algoritmically moves the coordinates to scan multiple regions   
        - as inputs we give lon, lat, folder name to export , spatial coverage of each image and cloud thresholds
        - we can target and download only the images that have a cloud score below a certain threshold
        - we can download hundreds of images (most of them cloud free) from a given location
    - we made another notebook called tif_pairwise_plotter. The notebook does the following:
        - imports image files from the download directory and converts them to numpy arrays
        - scales landsat and sentinel images seperately to RGB scale by linear transformation
        - plots the images pairwise 
        - it can filter out the images that are cloudy 
* W2D1 *
    - Second meeting with South Pole.
        - We showed them our image samples and got their confirmation to proceed with our image collection method.
        - We agreed that the time difference pair of images should not exceed 7 days
    - Updated our tif_exporter notebook to filter out image pairs that are more than 7 days apart
    - Updated our plotter notebook to filter out cloudy and bad-pixel-containing  images. 
    - Started downloading image pairs from Amazons, Madagascar, Honduras, Angola/DRC, and California.
* W2D2 *
    - Data collection completed from different regions : Amazon, Angola, Australia, California, Honduras, Madagaskar, Portugal
    - Images are selected from areas of deforestation
    - tif_exporter_moving_square notebook is used to download images from a selected region
    - tif_filtering_post_download notebook is used to algoritmically filter out cloudy and partially black out images
    - tif_manual_filtering notebook is used to manually filter out cloudy and non-informative images
* W2D3 *
    - continued data sourcing, with focus on manual filtering of auto-filtered datasets
    - background reading on super-resolution models 
* W2D4 *
    - Explored the repository https://github.com/idealo/image-super-resolution 
    - Updated ISR code to work with GEOTIFF images, with appropriate RGB scaling for sentinel/landsat 
    - Updated ISR code to run predictions on GEOTIFF data in batch mode. 
    - Ran test predictions using pre-trained weights of the RDN model on the California dataset to get pre-finetuning baseline
* W3D1 *
    - Train - Validation split 
        - Moved all 1'554 images in a single folder , 
        - Ran the notebook train_val_split.ipynb
        - Random shuffling of images
        - Train size: 660 pairs
        - Validation size : 117 pairs  
    - Image scaling
        - discovered that dataset contains small (1-3 pixel) mismatches in size 
        - discovered that the pre-trained weights of the ISR models are only applicable to 2x (for RDN) and 4x (for SRGAN) scaled images, while the difference between sentinel and landsat is 3x. Thus, available pre-trained weights cannot be used. 
    - RRDN and RDN model training on DIV2k-3x
        - Downloaded DIV2k data with 3 times downscaled LR images to use for training RRDN and RDN models of ISR repository
        - Trained 50 epochs of RRDN with only generator PSNR 
        - Trained RDN large and small models on DIV2k 
* W3D2 *
    - RRDN and RDN model training on DIV2k 
        - Trained RRDN for another 50 epochs with discriminator and feature extractor
    - Metrics
        - implemented sewar library for computing similarity metrics between images 
    - Meeting with SouthPole
* W3D3 *
    - image pre-processing/standardization: ran conversion of raw tiff images into images scaled and resized to 265 px (landsat) and 795 px (sentinel)
    - RRDN model training on DIV2k+landsat-sentinel dataset (on raw images, pre-processed within modified ISR package)
        -hyperparameter exploration
* W3D4 *
    - RRDN and RDN model training on DIV2k+landsat-sentinel dataset (on raw images, pre-processed within modified ISR package)
        -hyperparameter exploration
* W3D5 *
    - training RRDN from scratch on preprocessed images (first generator loss only, then other losses added)
    - ran RDN model predictions: 4 weights for small RDN, 4 weights for large RDN
        1 - no pre-traning on DIV2K
        2 - pretrained 100 epoch
        3 - pretrained 150 epoch
        4 - pretrained 200 epoch
    - Metrics
        - implemented saving similarity metrics between images (model's predictions vs sentinel) in a dataframe (csv)
        - implemented visualisations (histograms with summary metrics (mean, median)) per dataframe 
--------------------------------------------------------------------------------------

