import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import keras
from keras.models import Sequential # type: ignore
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Input, GlobalAveragePooling1D, BatchNormalization, Activation # type: ignore
from keras.utils import to_categorical # type: ignore


def plot_graphs(history) -> None:
    '''
    Plots 3 graphs using a sequential model's history: Training vs. Validation Loss, Training vs. Validation Accuracy,
    and Training vs. Validation AUC.
    
    Args:
        history: A history object (i.e., a record of training and validation metrics)
    
    Returns:
        None:
    '''
    # Plots training loss versus validation loss
    plt.plot(history.epoch, history.history['loss'], 'b', label='Training Loss')
    plt.plot(history.epoch, history.history['val_loss'], 'g', label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()
    
    # Plots training accuracy versus validation accuracy
    plt.plot(history.epoch, history.history['Accuracy'], 'b', label='Training Accuracy')
    plt.plot(history.epoch, history.history['val_Accuracy'], 'g', label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()
    
    # Plots training area under curve versus validation area under curve
    plt.plot(history.epoch, history.history['AUC'], 'b', label='Training AUC')
    plt.plot(history.epoch, history.history['val_AUC'], 'g', label='Validation AUC')
    plt.legend()
    plt.title('AUC')
    plt.show()
    
    
def evaluate_model(model: keras.Sequential, test_data: np.ndarray, test_labels: np.ndarray) -> list[float]:
    '''
    Evaluates a pre-trained model on the test data provided.
    
    Args:
        model: The pre-trained model to be evaluated
        test_data: A matrix of shape (n_samples, n_time_points)
        test_labels: The labels corresponding the the test/validation dataset
        
    Returns:
        list: A two-element list containing the model's loss and accuracy scores
    '''
    score = model.evaluate(test_data, test_labels, verbose=1)
    
    print(f'\nLoss on test data: {score[0]:.2%}')
    print(f'\nAccuracy on test data: {score[1]:.2%}')
    
    return score


def model_predict(model: keras.Sequential, test_data: np.ndarray, test_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Generates predictions using the inputted model and test data. Returns the true class labels along with the predicted 
    labels for the test data.
    
    Args:
        model: The pre-trained model that will be generating predictions
        test_data: A matrix of shape: (n_samples, n_time_points)
        test_labels: The labels corresponding to the test/validation dataset
        
    Returns:
        tuple: A two-element tuple containing the ground truth labels and the model's predicted labels
    '''
    y_test_pred = model.predict(test_data)
    
    # Taking the class with the highest probability based off the model's predictions
    y_hat = np.argmax(y_test_pred, axis=1)
    
    y_test = np.argmax(test_labels, axis=1)
    
    return (y_test, y_hat)


def show_confusion_matrix(test_labels: np.ndarray, prediction_labels: np.ndarray, classes: np.ndarray):
    '''
    Visualizes the performance of a model via a confusion matrix.
    
    Args:
        test_labels: The ground truth labels
        prediction_labels: The generated labels predicted by a pre-trained model
        classes: The labels of the unique classes within the dataset (i.e., [CD, HYP, MI, NORM, STTC])
        
    Returns:
        Figure: The confusion matrix figure  
    '''
    matrix = confusion_matrix(test_labels, prediction_labels)
    
    plt.figure(figsize=(6, 4))
    sns_hm=sns.heatmap(matrix,
                         cmap="YlGnBu",
                         linecolor='white',
                         linewidths=1,
                         xticklabels=classes,
                         yticklabels=classes,
                         annot=True,
                         fmt="d")
    
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    figure=sns_hm.get_figure()  
    plt.show()
    
    return figure


def generate_default_1D_model():
    '''
    Generates and returns a Keras sequential model with 14 convolutional layers. Each layer consists of ReLU 
    activation while the output layer utilizes softmax activation to predict the the most probable superclasses 
    present within adata point.
    '''
    model = Sequential([Input(shape=(400,1))])
    model.add(Conv1D(32, 18, name='conv0', activation='relu'))
    model.add(Conv1D(32, 18, name='conv1', activation='relu'))
    model.add(Conv1D(64, 18, name='conv2', activation='relu'))
    model.add(Conv1D(64, 18, name='conv3', activation='relu'))
    model.add(Conv1D(128, 18, name='conv4', activation='relu'))
    model.add(Conv1D(128, 18, name='conv5', activation='relu'))
    model.add(Conv1D(256, 18, name='conv6', activation='relu'))
    model.add(Conv1D(256, 18, name='conv7', activation='relu'))
    model.add(MaxPooling1D(3, name='max1'))
    model.add(Conv1D(32, 18, name='conv8', activation='relu'))
    model.add(Conv1D(32, 18, name='conv9', activation='relu'))
    model.add(Conv1D(64, 18, name='conv10', activation='relu'))
    model.add(Conv1D(64, 18, name='conv11', activation='relu'))
    model.add(Conv1D(128, 18, name='conv12', activation='relu'))
    model.add(Conv1D(256, 3, name='conv13', activation='relu'))
    model.add(GlobalAveragePooling1D(name='gap1'))
    model.add(Dropout(0.5, name='drop1'))
    model.add(Dense(5, name='dense1', activation='softmax'))
    return model


def generate_article_1D_model():
    '''
    Generates and returns a Keras sequential CNN model with 7 convolutional blocks. The initial 6 blocks consist of 1D 
    convolutional layers with ReLU activation, batch normalization, max pooling, and dropout layers. The final block
    implements a 1D convolutional layer, however, it performs global average pooling and has a larger droput value.
    
    Args:
        None:
        
    Returns:
        keras.Sequential: The model generated
    '''
    model = Sequential([Input(shape=(5000,1))])
    
    # Convolutional Block 1
    model.add(Conv1D(32, 3, name='conv1'))
    model.add(BatchNormalization(name='bn1'))
    model.add(Activation('relu', name='relu1'))
    model.add(MaxPooling1D(2, name='max1'))
    model.add(Dropout(0.2, name='drop1'))
    
    # Convolutional Block 2
    model.add(Conv1D(64, 3, name='conv2'))
    model.add(BatchNormalization(name='bn2'))
    model.add(Activation('relu', name='relu2'))
    model.add(MaxPooling1D(2, name='max2'))
    model.add(Dropout(0.2, name='drop2'))
    
    # Convolutional Block 3
    model.add(Conv1D(128, 3, name='conv3'))
    model.add(BatchNormalization(name='bn3'))
    model.add(Activation('relu', name='relu3'))
    model.add(MaxPooling1D(2, name='max3'))
    model.add(Dropout(0.2, name='drop3'))
    
    # Convolutional Block 4
    model.add(Conv1D(256, 3, name='conv4'))
    model.add(BatchNormalization(name='bn4'))
    model.add(Activation('relu', name='relu4'))
    model.add(MaxPooling1D(2, name='max4'))
    model.add(Dropout(0.2, name='drop4'))
    
    # Convolutional Block 5
    model.add(Conv1D(512, 3, name='conv5'))
    model.add(BatchNormalization(name='bn5'))
    model.add(Activation('relu', name='relu5'))
    model.add(MaxPooling1D(2, name='max5'))
    model.add(Dropout(0.2, name='drop5'))
    
    # Convolutional Block 6
    model.add(Conv1D(1024, 3, name='conv6'))
    model.add(BatchNormalization(name='bn6'))
    model.add(Activation('relu', name='relu6'))
    model.add(MaxPooling1D(2, name='max6'))
    model.add(Dropout(0.2, name='drop6'))
    
    # Convolutional Block 7
    model.add(Conv1D(2048, 3, name='conv7'))
    model.add(BatchNormalization(name='bn7'))
    model.add(Activation('relu', name='relu7'))
    model.add(GlobalAveragePooling1D(name='gap1'))
    model.add(Dropout(0.5, name='drop7'))
    
    # Dense Layer
    model.add(Dense(5, name='dense1', activation='softmax'))
    
    return model


def mc_label_encoding(train_labels: pd.Series, test_labels: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelBinarizer]:
    '''
    Transforms the labels for both the training and testing datasets into one-hot binary matrices.
    
    Args:
        train_labels: The labels corresponding to the training dataset
        test_labels: The labels corresponding the the testing/validation dataset
        
    Returns:
        tuple: A four-element tuple containing the transformed training and testing labels, unique classes, and a fitted label binarizer
    '''
    label_binarizer = LabelBinarizer()
    
    # Encodes labels
    y_train = label_binarizer.fit_transform(train_labels)
    y_test = label_binarizer.transform(test_labels)
    
    # Retrieving classes
    classes = label_binarizer.classes_
    
    return (y_train, y_test, classes, label_binarizer)