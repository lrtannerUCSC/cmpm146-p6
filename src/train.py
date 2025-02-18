import numpy as np
from preprocess import get_datasets
from models.basic_model import BasicModel
from models.model import Model
from config import image_size
import matplotlib.pyplot as plt
import time
import os
import sys

input_shape = (image_size[0], image_size[1], 3)
categories_count = 3

models = {
    'basic_model': BasicModel,
}

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize = (24, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()
    
def get_unique_filename(directory, base_filename):
    filename = base_filename
    counter = 1
    while os.path.exists(os.path.join(directory, filename)):
        filename = f"{base_filename}_{counter}.txt"
        counter += 1
    return filename


if __name__ == "__main__":
    # if you want to load your model later, you can use:
    # model = Model.load_model("name_of_your_model.keras")
    # to load your history and plot it again, you can use:
    # history = np.load('results/name_of_your_model.npy',allow_pickle='TRUE').item()
    # plot_history(history)
    # 
    # Your code should change the number of epochs
    epochs = 10
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_datasets()
    name = 'basic_model'
    model_class = models[name]
    print('* Training {} for {} epochs'.format(name, epochs))
    model = model_class(input_shape, categories_count)
    
    model.print_summary()

    # Create summary directory if it doesn't exist
    summary_dir = os.path.join(os.path.dirname(__file__), 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Get unique filename
    base_filename = 'model_summary'
    summary_file = get_unique_filename(summary_dir, base_filename)
    
    # Redirect stdout to a file for model summary
    with open(os.path.join(summary_dir, summary_file), 'w') as f:
        sys.stdout = f
        model.print_summary()
        sys.stdout = sys.__stdout__
    
    history = model.train_model(train_dataset, validation_dataset, epochs)
    print('* Evaluating {}'.format(name))
    model.evaluate(test_dataset)
    print('* Confusion Matrix for {}'.format(name))
    print(model.get_confusion_matrix(test_dataset))
    model_name = '{}_{}_epochs_timestamp_{}'.format(name, epochs, int(time.time()))
    filename = 'results/{}.keras'.format(model_name)
    model.save_model(filename)
    np.save('results/{}.npy'.format(model_name), history)
    print('* Model saved as {}'.format(filename))
    plot_history(history)
