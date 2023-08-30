import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,  roc_curve, auc, roc_auc_score
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = "json_files/features.json"

dropout_prob = 0.3
learning_rate=0.0001
epochs = 30

class_labels = [
    "pop",
    "metal",
    "disco",
    "blues",
    "reggae",
    "classical",
    "rock",
    "hiphop",
    "country",
    "jazz"
    ]  

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    #convert lists into numpy arrays    
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    
    return x, y

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epochs")
    axs[0].legend(loc="lower right")
    axs[0].set_title("ACCURACY")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epochs")
    axs[1].legend(loc="upper right")
    axs[1].set_title("ERROR")

    plt.show()

def prepare_datasets(test_size, validation_size):
    
    #load data
    x, y = load_data(DATASET_PATH)
    
    # create the train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    
    # create the train/validation spplit
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size)
 
    return x_train, x_test, x_val, y_train, y_test, y_val

def plot_confusion(y_test, y_pred_labels):
    
    cm = confusion_matrix(y_test, y_pred_labels)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_roc(y_test, y_pred):
    
        #ROC-AUC
    num_classes = len(np.unique(y_test))
    fpr = {}
    tpr = {}
    roc_auc = {}
        
    plt.figure(figsize=(12, 8))  # Create a single figure for all plots
    
    for i in range(num_classes):
        binary_labels = (y_test == i).astype(int)  # OvR strategy
        fpr[i], tpr[i], _ = roc_curve(binary_labels, y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='ROC curve (AUC = {:.2f}) for {}'.format(roc_auc[i], class_labels[i]))

    # Plot ROC curves for each genre with different color
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for LSTM')
    plt.legend(loc="lower right")
    plt.show()
        
def build_model(input_shape):
    
    #create model 
    model = tf.keras.Sequential()
    
    # 2 lstm layers
    model.add(tf.keras.layers.LSTM(64, input_shape = input_shape, return_sequences=True ))
    model.add(tf.keras.layers.LSTM(64))
    
    #dense layer
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_prob))
    
    # output layer   
    model.add(tf.keras.layers.Dense(10, activation='softmax'))   # each neuron represent different genre
    
    return model
        
if __name__ == "__main__":
    
    #create train, validation and test sets (we split in 3)
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_datasets(0.25,0.2)  
        # 25% goes to test 
        # From 75% and remaining 20% goes to validation
    
    #create the CNN net
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_model(input_shape)
    
    #compile the network
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss= "sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    model.summary()
    
    #train the CNN
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=epochs)
    
    #evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    
    y_pred = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    #plot various metrics
    plot_history(history)
    plot_confusion(y_test, y_pred_labels)
    plot_roc(y_test, y_pred)
    
    print("ROC AUC OVR: {} \n".format(roc_auc_score(y_test,y_pred,multi_class="ovr")))
    
    print(classification_report(y_test, y_pred_labels))