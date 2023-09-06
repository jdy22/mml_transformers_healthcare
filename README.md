# Context-Aware Transformers for Medical Image Segmentation

This repository contains all of the code to train and evaluate the models investigated in the individual project 'Context-Aware Transformers for Medical Image Segmentation'. The code is based on the original UNETR code, which is available at https://monai.io/research/unetr and cloned in the 'monai_research_contributions_main' folder of this repository. To carry out the experiments in this project, the original UNETR code has been adapted from 3D to 2D and various options for multi-modal learning have been added.

## Running the code

### Getting started

- Clone this repository and install all of the libraries in requirements.txt. 
- Download the AMOS dataset, which is available at https://zenodo.org/record/7155725#.Y0OOCOxBztM, and save it in the repository folder. The dataset must be contained in a folder named 'amos22'.

### Training a model

A model can be trained by running the main.py file. The value of the argument 'additional_information' in line 106 can be changed to specify different methods of adding contextual information to the model. To train each of the models discussed in the project report, the following values of 'additional_information' should be set:

| Model number | Type of information | Embeddings used | Fusion method | Value of 'additional_information' |
| -------- | ------- | ------- | ------- | ------- |
| 1 | Organ | Learnable | Early concatenation | "organ" |
| 2 | Organ | CLIP | Early concatenation | "clip_early" |
| 3a | Organ | Learnable | Intermediate concatenation | "organ_inter" |
| 3b | Organ | Learnable | Intermediate concatenation | "organ_inter2" |
| 3c | Organ | Learnable | Intermediate concatenation | "organ_inter3" |
| 4 | Organ | Learnable | Late concatenation | "organ_late" |
| 5 | Organ | CLIP | CLIP-driven inspired | "clip_late" |
| 6a | Image modality | Learnable | Early concatenation | "modality_concat" |
| 6b | Image modality | Learnable | Early concatenation | "modality_concat2" |
| 6c | Image modality | Learnable | Early summation | "modality_add" |
| 7a | Image modality | - | Separate decoders and output layers | "modality_decoder" |
| 7b | Image modality | - | Separate decoders and output layers | "modality_decoder_pretrained" |
| 8 | - | - | Joint classification and segmentation | "organ_classif_inter3" |
| Baseline | - | - | - | "none" |

A training log and the best model will be saved to the log directory specified by the argument 'logdir' in line 40.

### Evaluating a model

Once trained, a model can be evaluated using the val_dice_dist.py file in the post_processing folder. This should be run from the root folder of the directory. The 'pretrained_dir' argument in line 32 should be changed to the directory of the model to be evaluated, and the 'additional_information' argument in line 78 should be changed to specify the method of adding contextual information, as per the table above. The code will load the saved model and evaluate it on the images in the official validation set. It will report the Dice score, HD95 score and missed predictions for both CT and MRI images as averages over all of the organs as well as per organ.