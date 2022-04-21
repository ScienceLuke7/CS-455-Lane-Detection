# CS-455-Lane-Detection
To train the model:
    1. Follow the instructions file in _datasets to prepare the data first
    2. Install all dependencies listed in requirements.txt; `conda install requirements.txt` (may need to install some manually so search up `anaconda #packagename` in Google)
    3. Execute the `data_creater.py` file. This should create a `train_labels` folder
    4. Execute `model_trainer.py`. This will save the model under a folder called `saved_cnn_model`

To evaluate the model:
    1. Execute `model_evaluator.py` and an image window should appear along with a metric in the terminal