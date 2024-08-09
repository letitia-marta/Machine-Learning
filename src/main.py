import numpy as np
import tensorflow as tf
import os
import pandas as pd
import cv2 as cv
import timeit
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Rescaling, AveragePooling2D, Dropout
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

TRAIN_PATH = 'D:/BNN/Proiect/datasets/GTSRB_dataset/Train/'
TEST_PATH = 'D:/BNN/Proiect/datasets/GTSRB_dataset/'
ANNOTATIONS_FILE = 'D:/BNN/Proiect/datasets/GTSRB_dataset/Test.csv'
CLASSES = 43
IMG_SIZE = (32, 32)

def load_images (path, classes, img_size):
    images = []
    labels = []
    for i in range(classes):
        class_path = os.path.join(path, str(i))
        img_folder = os.listdir(class_path)
        for img_name in img_folder:
            try:
                image_path = os.path.join(class_path, img_name)
                image = cv.imread(image_path)
                if image is None:
                    print(f"Warning: Could not load image {image_path}")
                    continue
                image = cv.resize(image, img_size)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                images.append(image)
                label = np.zeros(classes)
                label[i] = 1.0
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
                pass
    return np.array(images) / 255.0, np.array(labels)

def load_test_images (annotations_file, test_path, classes, img_size):
    images = []
    labels = []
    annotations = pd.read_csv(annotations_file)
    for index, row in annotations.iterrows():
        try:
            image_path = os.path.join(test_path, row['Path'])
            image = cv.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue
            image = cv.resize(image, img_size)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            images.append(image)
            label = np.zeros(classes)
            label[int(row['ClassId'])] = 1.0
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {row['Path']}: {e}")
            pass
    return np.array(images) / 255.0, np.array(labels)

def LeNet1():
    model = Sequential([
        Rescaling(1, input_shape = (28, 28, 3)),
        Conv2D(filters = 4, kernel_size = (5, 5), activation = 'relu'),
        AveragePooling2D(pool_size = (2, 2)),
        Conv2D(filters = 12, kernel_size = (5, 5), activation = 'relu'),
        AveragePooling2D(pool_size = (2, 2)),
        Flatten(),
        Dense(units = 43, activation = 'softmax')
    ])
    return model

def LeNet5():
    model = Sequential([
        Rescaling(1, input_shape = (32, 32, 3)),
        Conv2D(filters = 6, kernel_size = (5, 5), activation = 'tanh'),
        AveragePooling2D(pool_size = (2, 2)),
        Conv2D(filters = 16, kernel_size = (5, 5), activation = 'tanh'),
        AveragePooling2D(pool_size = (2, 2)),
        Flatten(),
        Dense(units = 400, activation = 'tanh'),
        Dense(units = 84, activation = 'tanh'),
        Dense(units = 43, activation = 'softmax')
    ])
    return model

def save_model_summary (model, filename):
    with open(filename, 'w') as f:
        model.summary(print_fn = lambda x: f.write(x + '\n'))

def train_model (model, X_train, y_train, X_val, y_val, X_test, y_test, epochs = 50):
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    class TestAccuracyCallback (tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs = None):
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 0)
            logs['test_accuracy'] = test_acc
            logs['test_loss'] = test_loss

    test_acc_callback = TestAccuracyCallback()

    history = model.fit(X_train, y_train, epochs = epochs, validation_data = (X_val, y_val), batch_size = 32, callbacks = [test_acc_callback])
    
    return history

def evaluate_model (model, X_val, y_val, X_test, y_test):
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose = 2)
    print(f'\nValidation accuracy: {val_acc * 100:.2f}%')
    print(f'Validation loss: {val_loss:.4f}')

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 2)
    print(f'\nTest accuracy: {test_acc * 100:.2f}%')
    print(f'Test loss: {test_loss:.4f}')

    return val_acc, test_acc

def plot_history (history):
    plt.figure(0)
    plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
    plt.plot(history.history['test_accuracy'], label = 'Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc = 'lower right')
    plt.savefig('accuracy_plot.png')

    plt.figure(1)
    plt.plot(history.history['val_loss'], label = 'Validation Loss')
    plt.plot(history.history['test_loss'], label = 'Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 0.2])
    plt.legend(loc = 'lower right')
    plt.savefig('loss_plot.png')

    plt.show()

def plot_confusion_matrix (y_true, y_pred, classes):
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis = 1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(conf_matrix_normalized, display_labels = range(classes))

    plt.figure(figsize = (20, 20))
    disp.plot(cmap = plt.cm.Blues, values_format = '.2f', ax = plt.gca())
    plt.title('Confusion Matrix', fontsize = 24)
    plt.xticks(fontsize = 5)
    plt.yticks(fontsize = 5)
    plt.gca().images[-1].colorbar.ax.tick_params(labelsize = 16)

    for text in disp.text_.ravel():
        text.set_fontsize(5)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_test_predictions (X_test, preds, y_test, start_index):
    plt.figure(figsize = (12, 12))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        pred = np.argmax(preds[start_index + i])
        gt = np.argmax(y_test[start_index + i])
        col = 'g' if pred == gt else 'r'
        plt.xlabel(f'i={start_index + i}, pred={pred}, gt={gt}', color = col)
        plt.imshow(X_test[start_index + i])
    plt.savefig('test_predictions.png')
    plt.show()

def plot_roc_curve (y_true_binarized, preds_binarized):
    fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), preds_binarized.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize = (10, 8))
    plt.plot(fpr, tpr, color = 'blue', lw = 2, label = f'Micro-average ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw = 2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 14)
    plt.ylabel('True Positive Rate', fontsize = 14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize = 16)
    plt.legend(loc = 'lower right', fontsize = 12)
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.show()

def main():
    train_images, train_labels = load_images(TRAIN_PATH, CLASSES, IMG_SIZE)
    test_images, test_labels = load_test_images(ANNOTATIONS_FILE, TEST_PATH, CLASSES, IMG_SIZE)
    
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size = 0.2, random_state = 42)
    X_test, y_test = test_images, test_labels

    model = LeNet5()
    save_model_summary(model, 'LeNet5_architecture.txt')

    start = timeit.default_timer()
    history = train_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
    end = timeit.default_timer()
    print(f"Training time: {end - start:.2f} seconds")
    
    evaluate_model(model, X_val, y_val, X_test, y_test)
    plot_history(history)
    
    preds = model.predict(X_test)
    y_true = np.argmax(y_test, axis = 1)
    y_pred = np.argmax(preds, axis = 1)
    
    plot_confusion_matrix(y_true, y_pred, CLASSES)
    plot_test_predictions(X_test, preds, y_test, random.randint(0, len(X_test) - 25))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names = [str(i) for i in range(CLASSES)]))
    
    y_true_binarized = label_binarize(y_true, classes = range(CLASSES))
    plot_roc_curve(y_true_binarized, preds)

if __name__ == "__main__":
    main()
