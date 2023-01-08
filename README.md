# Couch Material Changer
Uses image segmentation and style transfer to change the materials of a couch. Also includes the option to only change the color of the couch. The segmentation model reached a validation loss of 0.2 and a validation binary IOU of 0.8.

## Input to Output Diagram
<img src="CouchMaterialChangerExample.png" width="600" height="1100" />

## Dataset
Uses the 2017 COCO dataset. The segmentation model is pretrained with the MobileNetV2 architecture. The style transfer model is pretrained with the VGG-19 architecture.

## Usage
Download the files and run the "test_model.py" file.
