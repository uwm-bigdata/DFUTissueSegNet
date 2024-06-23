# DFUTissueSegNet
### Instruction
Step-1: Run `Codes/supervised_training.ipynb`. This code is to train the hybrid model using labeled data. <br>
Step-2: Run `Codes/pseudo_label_generation_phase1.ipynb`. This code predictions (pseudo labels) from unsupervised images (no labels) using the trained model obtained after the supervised phase. <br>
Step-3: Run `Codes/semisupervised_training_phase1.ipynb`. This code trains the model 5 times. Each time, it randomly takes 50 unsupervised images (and their corresponding pseudo labels) along with 78 labeled images to train the model. It also reports the best validation losses for all runs. <br>
