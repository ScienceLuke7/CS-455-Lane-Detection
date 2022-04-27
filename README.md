# CS-455-Lane-Detection
To train the model:
    1. Follow the instructions file in _datasets to prepare the data first
    2. Install all dependencies listed in requirements.txt; `conda create -n lane-detection --file requirements.txt`
    3. Execute the `data_creater.py` file. This should create a `train_labels` folder
    4. Execute `model_trainer.py`. This will save the model under a folder called `saved_cnn_model`

To evaluate the model:
    1. Execute `model_evaluator.py` and an image window should appear along with a metric in the terminal

To host the Angular GUI:
    1. Install npm (Node Package Manager)
    2. In a shell type `npm install -g @angular/cli`
    3. Navigate into the GUI folder and type `ng serve`. This will start the build and hosting process

To host the API:
    1. Install the Python package, Flask, by using either Anaconda or pip.
    2. In a shell or IDE, execute `api.py`
