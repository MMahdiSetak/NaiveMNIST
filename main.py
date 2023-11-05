import numpy as np
from os.path import join
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage.measure import block_reduce
from scipy.ndimage import correlate
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from MNISTDataLoader import MnistDataloader

input_path = './MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                   test_labels_filepath)
(x_train_raw, y_train), (x_test_raw, y_test) = mnist_dataloader.load_data()

# Normalize raw data
x_train_raw = np.array(x_train_raw) / 255
x_test_raw = np.array(x_test_raw) / 255

results_rows = []
results_columns = ('Feature Set', 'No. Features', 'Accuracy (Gaussian)', 'Accuracy (Multinomial)')


def flat_feature(img):
    """Flatten the image to a 1D array."""
    return img.flatten()


def row_sum_feature(img):
    """Calculate the sum of each row and normalize."""
    return np.sum(img, axis=1) / 28


def col_sum_feature(img):
    """Calculate the sum of each column and normalize."""
    return np.sum(img, axis=0) / 28


def sum_kernel_feature(image):
    """Apply window summation to the image."""
    kernel = np.ones((7, 7))
    # Perform convolution with zero padding (which is default)
    result = correlate(image, kernel) / kernel.sum()
    # Trim the edges to get a valid output and flatten the result
    return result[3:-3, 3:-3].flatten()


def sum_pooling_feature(img):
    """Apply sum pooling to the image."""
    return block_reduce(img, block_size=2, func=np.sum).flatten() / 4


def max_pooling_feature(img):
    """Apply max pooling to the image."""
    return block_reduce(img, block_size=2, func=np.max).flatten()


def hog_feature(img):
    """Extract Histogram of Oriented Gradients (HOG) features from the image."""
    return hog(img.reshape((28, 28)), orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=False)


def apply_features(img, *func):
    """Apply multiple feature extraction functions to the image."""
    ret = []
    for f in func:
        ret.extend(f(img))
    return ret


def prepare_train_test(train, test, *func):
    """Prepare training and testing sets by applying feature extraction."""
    prepared_train = Parallel(n_jobs=-1)(delayed(apply_features)(img, *func) for img in train)
    prepared_test = Parallel(n_jobs=-1)(delayed(apply_features)(img, *func) for img in test)
    return prepared_train, prepared_test


def train(features_train, label_train, features_test, label_test, feature_set):
    """Train Naive Bayes models and record their accuracy."""
    gnb = GaussianNB()
    mnb = MultinomialNB()
    gnb.fit(features_train, label_train)
    mnb.fit(features_train, label_train)
    y_pred_g = gnb.predict(features_test)
    y_pred_m = mnb.predict(features_test)
    accuracy_g = accuracy_score(label_test, y_pred_g)
    accuracy_m = accuracy_score(label_test, y_pred_m)
    results_rows.append([feature_set, len(features_train[0]), accuracy_g, accuracy_m])


x_train, x_test = prepare_train_test(x_train_raw, x_test_raw, flat_feature)
train(x_train, y_train, x_test, y_test, "Raw Data")

x_train, x_test = prepare_train_test(x_train_raw, x_test_raw, row_sum_feature)
train(x_train, y_train, x_test, y_test, "Sum of Rows")

x_train, x_test = prepare_train_test(x_train_raw, x_test_raw, col_sum_feature)
train(x_train, y_train, x_test, y_test, "Sum of Columns")

x_train, x_test = prepare_train_test(x_train_raw, x_test_raw, col_sum_feature, row_sum_feature)
train(x_train, y_train, x_test, y_test, "Sum of Rows and Columns")

x_train, x_test = prepare_train_test(x_train_raw, x_test_raw, sum_kernel_feature)
train(x_train, y_train, x_test, y_test, "Kernel Sum")

x_train, x_test = prepare_train_test(x_train_raw, x_test_raw, sum_pooling_feature)
train(x_train, y_train, x_test, y_test, "Sum Pooling")

x_train, x_test = prepare_train_test(x_train_raw, x_test_raw, max_pooling_feature)
train(x_train, y_train, x_test, y_test, "Max Pooling")

x_train, x_test = prepare_train_test(x_train_raw, x_test_raw, row_sum_feature, col_sum_feature, max_pooling_feature)
train(x_train, y_train, x_test, y_test, "Sum of Rows and Columns + Max Pooling")

x_train, x_test = prepare_train_test(x_train_raw, x_test_raw, hog_feature)
train(x_train, y_train, x_test, y_test, "HOG")

pca = PCA(n_components=64)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
m = np.min([x_train.min(), x_test.min()])
x_train -= m
x_test -= m
train(x_train, y_train, x_test, y_test, "HOG with PCA")

x_train, x_test = prepare_train_test(x_train_raw, x_test_raw, hog_feature, col_sum_feature, row_sum_feature)
train(x_train, y_train, x_test, y_test, "HOG + Column & Row Sum")

pca = PCA(n_components=128)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
m = np.min([x_train.min(), x_test.min()])
x_train -= m
x_test -= m
train(x_train, y_train, x_test, y_test, "HOG + Column & Row Sum with PCA")

fig, ax = plt.subplots(figsize=(11, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results_rows, colLabels=results_columns, loc='center', cellLoc='center')
table.set_fontsize(12)
table.scale(1.2, 1.2)
table.auto_set_column_width(col=list(range(len(results_columns))))
plt.show()
