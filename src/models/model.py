import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping


class Model:
    def __init__(self, input_shape, categories_count):
        self._define_model(input_shape, categories_count)
        self._compile_model()
        assert hasattr(self, "model"), "Model object does not include a keras model"

    def _define_model(self, input_shape, categories_count):
        raise Exception("define_model not implemented yet.")

    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    def train_model(self, train_dataset, validation_dataset, epochs):
        early_stopping = EarlyStopping( # early stopping to find best epoch count
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        # Train the model with early stopping
        history = self.model.fit(
            x=train_dataset,
            epochs=epochs,
            verbose="auto",
            validation_data=validation_dataset,
            callbacks=[early_stopping]
        )

        return history

    def save_model(self, filename):
        self.model.save(filename)

    @staticmethod
    def load_model(filename):
        return LoadedModel(filename)
    
    def evaluate(self, test_dataset):
        self.model.evaluate(
            x=test_dataset,
            verbose='auto',
        )
    
    def get_confusion_matrix(self, test_dataset):
        prediction = self.model.predict(test_dataset)
        labels = np.concatenate([y for x, y in test_dataset], axis=0)
        y_pred = np.argmax(prediction, axis=-1)
        y = np.argmax(labels, axis=-1)
        return confusion_matrix(y, y_pred)

    def print_summary(self):
        self.model.summary()
    
    def plot_model_shape(self):
        plot_model(self.model, show_shapes=True, to_file='test.png')

class LoadedModel(Model):
    def __init__(self, filename):
        self.model = load_model(filename)

    def _define_model(self, input_shape, categories_count):
        pass

    def _compile_model(self):
        pass