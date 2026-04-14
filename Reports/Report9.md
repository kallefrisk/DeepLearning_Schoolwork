## Report assignment 8, group 6

## ML


## Software Development

To make the model easy to train again and reuse, we created a clear and structured workflow.

The training is controlled by a dictionary of hyperparameters. This means we can change things like the model size or learning rate without changing the actual code. Because of this, it is easy to train the model again with new data or different parameters.

Every trained model can be saved to a file (candidates). This means we do not need to train the model again every time we want to use it. We also created a system to keep track of the best model, called the champion model. The best model is chosen based on how well it performs on validation data (using mean absolute error). It is also saved in a folder called champion.

If a new model performs better than the current champion, it replaces it. We also save information about the model, such as its parameters and performance in the folder metadata. This makes it possible to always go back and use the best model.

Overall, this makes the system easy to use, update, and improve without needing a lot of manual work.