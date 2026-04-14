## Report assignment 9, group 6

## ML

### Optimal model

### Data Loading & Preprocessing

The first step was loading the data, which was a relatively straightforward task. The dataset consisted of time-series pose estimation data from Kinect sensors, containing 13 joints with x and y coordinates (26 features total) as input, with the target being the next frame's joint positions (13 coordinates). Each file represented one complete squat sequence, with varying lengths across different recordings.

### Experiment Tracking with MLflow

The next step was setting up MLflow to log all runs, experiments, and model artifacts. Without this, it would have been impossible to systematically compare different hyperparameter configurations and track model performance over time.

### Hyperparameters

**Activation Functions:** ReLU and Leaky ReLU were tested. According to literature, ReLU is the recommended default choice, so this was used consistently.

**Optimizer:** Adam was used most of the time because, as with ReLU, it is widely regarded as the best default optimizer for most deep learning tasks.

**Network Architecture:** The size and number of hidden layers were the most critical hyperparameters. Configurations tested included [256,128,128,64,64] (5 layers), [256,128,128,64] (4 layers), [256,128,64] (3 layers), and [128,64] (2 layers) with more. LSTM, GRU, and Dense architectures were compared.

**Dropout Rate:** Dropout rates between 0.0 and 0.4 were tested. This regularization technique randomly turns off nodes during training to prevent overfitting.

**Weight Decay (L2 Regularization):** Weight decay values from 0 to 1e-5 were tested. When regularization was too large (1e-5), the model underfit.

**Learning Rate:** Learning rates of 0.0005 and 0.001 were tested, but no significant difference was observed.



**Early Stopping Patience:** This was implemented to stop training when validation loss stopped improving, preventing overfitting. 

### Errors 

**Padding Evaluation Mistake:** A critical oversight was that after padding the time-series data, we evaluated our models including the padded zeros. Since predicting zeros for padded frames is trivial, this artificially inflated our performance metrics.


### Future works

- **Domain Adaptation:** Possibility to transform Google MediaPipe pose estimations to Kinect-style data.



## Software Development

To make the model easy to train again and reuse, we created a clear and structured workflow.

The training is controlled by a dictionary of hyperparameters. This means we can change things like the model size or learning rate without changing the actual code. Because of this, it is easy to train the model again with new data or different parameters.

Every trained model can be saved to a file (candidates). This means we do not need to train the model again every time we want to use it. We also created a system to keep track of the best model, called the champion model. The best model is chosen based on how well it performs on validation data (using mean absolute error). It is also saved in a folder called champion.

If a new model performs better than the current champion, it replaces it. We also save information about the model, such as its parameters and performance in the folder metadata. This makes it possible to always go back and use the best model.

Overall, this makes the system easy to use, update, and improve without needing a lot of manual work.