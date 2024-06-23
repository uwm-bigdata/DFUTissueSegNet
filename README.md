# DFUTissueSegNet
### Instruction
* Step-1: Run `Codes/supervised_training.ipynb`. This code is to train the hybrid model using labeled data. 
* Step-2: Run `Codes/pseudo_label_generation_phase1.ipynb`. This code predictions (pseudo labels) from unsupervised images (no labels) using the trained model obtained after the supervised phase.
* Step-3: Run `Codes/semisupervised_training_phase1.ipynb`. This code trains the model 5 times. Each time, it randomly takes 50 unsupervised images (and their corresponding pseudo labels) along with 78 labeled images to train the model. It also reports the best validation losses for all runs. 
* Step-4: Run `Codes/pseudo_label_generation_phase2.ipynb`. This code creates pseudo labels during the semi-supervised phase. It first removes those 50 images from unsupervised images that generated the best model based on the validation loss. It then creates pseudo labels for the remaining unsupervised images.
* Step-5: Run `Codes/semisupervised_training_phase2.ipynb`. This code again trains the model 5 times. However, this time training images will be 78 (original labeled tissue data) + 50 (unsupervised images that generated the best model in the previous phase) + 50 (randomly picked images from the remaining unsupervised images along with their pseudo labels in this phase).
* Repeat Step-4 and Step-5 until there is not improvement in the validation loss.

