# M3D-Net
Research on the Detection Method of Flight Trainees’ Attention State Based on Multi-Modal Dynamic Depth Network

### Project Overview ###
This project is a classification model based on multimodal data (images, eye movement text data, and flight operation text data), aiming to classify the attention states of flight cadets in a simulated flight environment. The model mainly combines components such as MobileNet-V3, bidirectional LSTM, and ConvLSTM, and achieves the recognition of six different attention states (dangerous driving, distraction, normal level, high concentration, fatigue, yawning, etc.) through multimodal fusion.

### Project Structure ###
```
M3D-NET/
├── dealData
    ├── dealEyeTxtData.py       # Code for processing visual information file data
    ├── dealFlightTxtData.py    # Code for processing flight operation data information file data
    ├── dealImgData.py          # Code for processing sequential image file data
├── processData
    ├── dealEyeData             # Used to store processed visual information file data
    ├── dealFlightData          # Used to store processed flight operation file data
    ├── dealImgData             # Used to store processed sequential image file data
├── rowData
    ├── eye_data                # Used to store the original data of visual information files
    ├── flight_data             # Used to store the original data of flight operation files
    ├── image                   # Used to store the original data of sequential image files
├── train
    ├── classReport             # Used to store the classification reports of training
    ├── model                   # Used to store the trained model files
    ├── train_img               # Used to store the result images of training
    ├── config.py               # Stores all the configuration parameters of the project
    ├── dataset.py              # Defines the dataset class and related functions
    ├── model.py                # Defines the model class and related modules
    ├── trainer.py              # Defines the training and validation functions
    ├── utils.py                # Contains optimizer configuration and auxiliary functions
    └── main.py                 # The main function that calls other modules to complete the training and testing processes
```

### Main Functions of Core Code Files ###
### config.py ###
Parameter configuration: Defines the names of classification classes `class_names`, the number of classes `num_classes`, the length of the sequential sequence `sequence_length`, etc.
Device configuration: Sets the computing device `device` according to whether there is an available CUDA device.
Dataset split ratio: Defines the split ratios of the training set, validation set, and test set `split_ratios` and the random seed `random_seed`.
Data augmentation configuration: Defines data augmentation operations for the training set, validation set, and test set respectively, such as random cropping, flipping, color jittering, etc.

### dataset.py ###
Dataset class definition: The `TemporalDrivingDataset` class is used to load and process multimodal data, including image data, eye movement text data, and flight operation text data, and align them into samples.
File retrieval function: The `get_txt_files` function is used to retrieve all .txt files in a specified directory and its subdirectories.
Dataset split function: The `split_dataset` function splits the dataset into a training set, validation set, and test set according to the given ratio and random seed.

### model.py ###
Cross-modal attention module: The `CrossModalAttention` class implements the cross-modal attention mechanism for fusing image features and text features.
Depthwise separable ConvLSTM cell: The `DepthwiseSeparableConvLSTMCell` class defines a depthwise separable ConvLSTM cell for processing sequential data.
Complete model definition: The `StableCNNLSTM` class integrates the image processing branch (based on MobileNet V3), the text processing branch (bidirectional LSTM), cross-modal fusion, and sequential processing parts, and finally outputs the classification results.

### trainer.py ###
Training function: The `train_epoch` function is used to perform one training epoch, including forward propagation, loss calculation, backpropagation, and model parameter update, and statistics of training loss and accuracy.
Validation function: The `validate_epoch` function is used to evaluate the model on the validation set and calculate the validation loss and accuracy.

### utils.py ###
Optimizer configuration function: The `configure_optimizer` function sets different learning rates for different parts of the model and returns the optimizer and learning rate scheduler.
Auxiliary function: The `find_matching_indices` function is used to find the indices of equal elements in two lists.

### main.py ###
Data loading and splitting: Calls the functions in `dataset.py` to load the dataset and perform the split.
Model initialization and training: Initializes the model according to the available GPUs, calls the functions in `trainer.py` to perform training, and saves the best model.
Model evaluation: Loads the best model, tests it on the test set, calculates the accuracy, generates a classification report, and plots the confusion matrix.

### Running the Project ###
Ensure that all the required dependency libraries for the project are installed, such as torch, torchvision, tqdm, matplotlib, seaborn, sklearn, etc.
Modify the configuration parameters in `config.py` as needed.
```shell
cd M3D-NET/train/
python config.py
```
Run the `main.py` script to start training.
```shell
cd M3D-NET/train/
python main.py
```
