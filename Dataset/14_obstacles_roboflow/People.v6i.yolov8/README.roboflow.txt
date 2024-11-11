
People - v6 2024-06-05 9:59pm
==============================

This dataset was exported via roboflow.com on June 5, 2024 at 7:00 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 198 images.
People are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random shear of between -7° to +7° horizontally and -15° to +15° vertically
* Random exposure adjustment of between -10 and +10 percent
* Salt and pepper noise was applied to 1.8 percent of pixels


