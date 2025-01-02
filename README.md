# CS266 Final Project
This project is working ontop of an existing repo project from MohamadMerchant (https://github.com/MohamadMerchant/Voice-Authentication-and-Face-Recognition). Most of his code is concentrated within the main.ipynb notebook but is not modified completley to work as a front end for this project. Their code was a useful indication of how to work with audio files and for future steps, I do plan to retrofit my SVM classifier into his Python notebook. The main files of interest are the following
* voicepop.py: File that will do featuire extraction, such as extyracting pop noises and then outputting GFCC features, to end up training the SVM classifier. This is an implemnation of the Wang et al. paper (https://ieeexplore.ieee.org/document/8737422)
* voicepop_poison.py: same as the above file but with datapoisoning with label flipping and adding synthetic voicepop data to audio files
* svm_model_test.py: this will take the saved model and scaler outputted from the voicepop.py or voicepop_poison.py file and evaluate the model based on the evaluation dataset

Saved models can be used from the models_used directory and the filenames should indicate what type of model they are (1123_full_train_scaler/train.pkl indicate the full training with no poison). Make sure to replace the .pkl directory within svm_model_test in the function: process_audio_files

Install dependencies based on requirements.txt, the other one is based on the original project.

ASVSpoof 2019 Dataset can be accessed here: https://datashare.ed.ac.uk/handle/10283/3336