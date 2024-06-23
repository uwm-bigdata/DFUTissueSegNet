# DFUTissueSegNet

### Model
<div align="center">
	<img src="/Resources/model.png">
</div>

### DFUTissue Dataset
We created the DFUTissue dataset with the Advancing the Zenith of Healthcare (AZH) Wound Center in Milwaukee, WI, to segment granulation, fibrin, and callus tissues in diabetic foot ulcers (DFUs). The dataset includes 110 annotated foot ulcer images: 78 for training, 16 for validation, and 16 for testing, maintaining a 70:15:15 split. The images, sourced from the AZH Wound Center, were initially annotated by the Big Data Analytics and Visualization Lab at UWM and refined by AZH wound specialists. The dataset covers eight tissue categories: granulation, callus, fibrin, necrotic, eschar, neodermis, tendon, and dressing. Wound care specialists kept dry necrotic tissues but removed damp necrotic tissues. Annotating wound tissues is challenging due to less defined boundaries and irregular shapes. Table 1 shows the appearance count of each type of tissue.  The paper focuses on granulation, callus, and fibrin tissues due to the limited number of images for the other categories. 

<div align="center">
	<img src="/Resources/Table1.png">
</div>

### Instruction
* Step-1: Run `Codes/supervised_training.ipynb`. This code is to train the hybrid model using labeled data. 
* Step-2: Run `Codes/pseudo_label_generation_phase1.ipynb`. This code predictions (pseudo labels) from unsupervised images (no labels) using the trained model obtained after the supervised phase.
* Step-3: Run `Codes/semisupervised_training_phase1.ipynb`. This code trains the model 5 times. Each time, it randomly takes 50 unsupervised images (and their corresponding pseudo labels) along with 78 labeled images to train the model. It also reports the best validation losses for all runs. 
* Step-4: Run `Codes/pseudo_label_generation_phase2.ipynb`. This code creates pseudo labels during the semi-supervised phase. It first removes those 50 images from unsupervised images that generated the best model based on the validation loss. It then creates pseudo labels for the remaining unsupervised images.
* Step-5: Run `Codes/semisupervised_training_phase2.ipynb`. This code again trains the model 5 times. However, this time training images will be 78 (original labeled tissue data) + 50 (unsupervised images that generated the best model in the previous phase) + 50 (randomly picked images from the remaining unsupervised images along with their pseudo labels in this phase).
* Step-6: Repeat Step-4 and Step-5 until there is no improvement in the validation loss.

### Results for the DFUTissue dataset

<div align="center">
	<img src="/Resources/Table3_.png">
</div>
<br>

<div align="center">
	<img src="/Resources/Table4_.png">
</div>
<br>

<div align="center">
	<img src="/Resources/Table5.png">
</div>
<br>

### Results for the chronic wound dataset
<div align="center">
	<img src="/Resources/Table9.png">
</div>
<br>
