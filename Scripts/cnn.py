import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import keras
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense, BatchNormalization, Activation # type: ignore
from keras.models import Sequential # type: ignore


def plot_graphs(history, save_path: str) -> None:
    '''
    Plots 3 graphs using a sequential model's history: Training vs. Validation Loss, Training vs. Validation Accuracy,
    and Training vs. Validation AUC.
    
    Args:
        history: A history object (i.e., a record of training and validation metrics)
        save_path: The save path where the loss, accuracy, and AUC figures will be saved
    
    Returns:
        None:
    '''
    # Plots training loss versus validation loss
    plt.figure(figsize=(8,4))
    plt.plot(history.epoch, history.history['loss'], 'b', marker='.', label='Training Loss')
    plt.plot(history.epoch, history.history['val_loss'], 'g', marker='.', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(save_path, 'Loss.png'))
    plt.show()
    
    # Plots training accuracy versus validation accuracy
    plt.figure(figsize=(8,4))
    plt.plot(history.epoch, history.history['Accuracy'], 'b', marker='.', label='Training Accuracy')
    plt.plot(history.epoch, history.history['val_Accuracy'], 'g', marker='.', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(save_path, 'Accuracy.png'))
    plt.show()
    
    # Plots training area under curve versus validation area under curve
    plt.figure(figsize=(8,4))
    plt.plot(history.epoch, history.history['AUC'], 'b', marker='.', label='Training AUC')
    plt.plot(history.epoch, history.history['val_AUC'], 'g', marker='.', label='Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC Score')
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('AUC')
    plt.savefig(os.path.join(save_path, 'AUC.png'))
    plt.show()
    

def evaluate_model(model: keras.Sequential, test_data: np.ndarray, test_labels: np.ndarray) -> list[float]:
    '''
    Evaluates a pre-trained model on the test data provided.
    
    Args:
        model: The pre-trained model to be evaluated
        test_data: A matrix of shape (n_samples, n_time_points)
        test_labels: The labels corresponding the the test/validation dataset
        
    Returns:
        tuple: A two-element tuple, containing the model's loss and accuracy scores
    '''
    score = model.evaluate(test_data, test_labels, verbose=1)
    
    print(f'\nLoss on test data: {score[0]:.2%}')
    print(f'\nAccuracy on test data: {score[1]:.2%}')
    
    return score


def model_predict(model: keras.Sequential, test_data: np.ndarray, test_labels: np.ndarray, threshold: float=0.5) -> tuple[np.ndarray, np.ndarray]:
    '''
    Generates predictions using the inputted model and test data. The threshold specified will determine if a certain 
    superclass/label is present within a record.
    
    Args:
        model: The pre-trained model that will be generating the predictions
        test_data: A matrix of shape: (n_samples, n_time_points)
        test_labels: The labels corresponding to the test/validation dataset
        threshold: Determines the minimum value for considering a superclass present
        
    Returns:
        tuple: A two-element tuple containng the ground truth labels and the model's predicted labels
    '''
    y_test_pred = model.predict(test_data)
    
    # Converts probabilites into binary labels (i.e. either 0 or 1 if they meet the threshold)
    y_hat = (y_test_pred >= threshold).astype(int)
    
    # Ensures test data is in binary format
    y_test = test_labels.astype(int)
    
    return (y_test, y_hat)


def show_confusion_matrix(test_labels: np.ndarray, prediction_labels: np.ndarray, classes: np.ndarray) -> list:
    '''
    Visualizes the performance of a model via confusion matrices and heatmaps. A 2x2 confusion matrix is plotted and 
    displayed for each class.
    
    Args:
        test_labels: The ground truth labels
        prediction_labels: The generated labels predicted by a pre-trained model
        classes: The labels of the unique superclasses within the dataset (i.e., [CD, HYP, MI, NORM, STTC])
    
    Returns:
        list: A list containing figures corresponding to each superclass
    '''
    figures = []
    
    # Computes the confusion matrix for each individual class
    matrices = multilabel_confusion_matrix(test_labels, prediction_labels)
    
    for i, label in enumerate(classes):
        
        plt.figure(figsize=(6,4))
        
        sns_hm = sns.heatmap(matrices[i],
                            cmap='YlGnBu',
                            linecolor='white',
                            xticklabels=['Negative', 'Positive'],
                            yticklabels=['Negative', 'Positive'],
                            annot=True, # ensures values are on heatmap
                            fmt='d') # formats values as ints
        
        plt.title(f'{label} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        figure = sns_hm.get_figure()
        figures.append(figure)
        plt.show()
        
    return figures


def generate_default_1D_model():
    '''
    Generates and returns a Keras sequential model with 14 convolutional layers. Each layer consists of ReLU activation 
    while the output layer utilizes Sigmoid activation to predict the possible superclasses present within a data point.
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
    model.add(Dense(5, name='dense1', activation='sigmoid'))
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
    model.add(Dense(5, name='dense1', activation='sigmoid'))
    
    return model


def generate_12lead_model(input_shape: tuple=(1000,12), n_outputs: int=5):
    model = Sequential([
        Input(shape=input_shape),
        # Block 1: Per-lead temporal modeling
        Conv1D(36, 15, padding='same', groups=12, use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),
        
        # Learnable downsampling
        Conv1D(64, 7, strides=2, padding='same', use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),
        
        # Block 2: Cross-lead fusion
        Conv1D(128, 7, padding='same', use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),
        
        Conv1D(128, 5, strides=2, padding='same', use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),
        
        # Block 3: Deep abstraction
        Conv1D(256, 5, padding='same', use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),
        
        # Embedding head
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_outputs, activation='sigmoid')
    ])
    return model


def ml_label_encoding(train_labels: pd.Series | np.ndarray, test_labels: pd.Series | np.ndarray) -> tuple:
    '''
    Transforms the labels for both the training and testing datasets into multi-hot binary matrices.
    
    Args:
        train_labels: The labels corresponding to the training dataset
        test_labels: The labels corresponding the the testing/validation dataset
        
    Returns:
        tuple: A three-element tuple containing the transformed training and testing labels along with the unique classes
    '''
    mlb = MultiLabelBinarizer()
    
    # Encoding labels into binary matrices
    y_train = mlb.fit_transform(train_labels)
    y_test = mlb.transform(test_labels)
    
    # Storing the classes
    classes = mlb.classes_
    
    # Saves original index of labels if they were Series objects
    if isinstance(train_labels, pd.Series):
        y_train = pd.DataFrame(data=y_train, index=train_labels.index)
        
    if isinstance(test_labels, pd.Series):
        y_test = pd.DataFrame(data=y_test, index=test_labels.index)
    
    return (y_train, y_test, classes)