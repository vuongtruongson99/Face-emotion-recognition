from csv import writer
from config import emotion_config as config
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
import numpy as np

print("[INFO] Loading input data...")
print(config.INPUT_PATH)
f = open(config.INPUT_PATH)
f.__next__()    # __next__(): skip header of CSV file
(trainImages, trainLabels) = ([], [])
(valImages, valLabels) = ([], [])
(testImages, testLabel) = ([], [])

for row in f:
    (label, image, usage) = row.strip().split(",")
    label = int(label)

    # We are ignoring the "disgust" classes => 6 total class
    if config.NUM_CLASSES == 6:
        # Merge the "anger" and "disgust" classes
        if label == 1:
            label = 0

        if label > 0:
            label -= 1        

    # Now we have image - string of intergers
    #   + Need to split into a list
    #   + Convert to unsigned 8-bit interger
    #   + Reshape to 48x48 grayscale image
    image = np.array(image.split(" "), dtype="uint8")
    image = image.reshape((48, 48))

    # Check if image in a training dataset
    if usage == "Training":
        trainImages.append(image)
        trainLabels.append(label)

    # Check if is a validation image
    elif usage == "PrivateTest":
        valImages.append(image)
        valLabels.append(label)

    # Ortherwise, must be testing image
    else:
        testImages.append(image)
        testLabel.append(label)

datasets = [
    (trainImages, trainLabels, config.TRAIN_HDF5),
    (valImages, valLabels, config.VAL_HDF5),
    (testImages, testLabel, config.TEST_HDF5)
]

for (images, labels, outputPath) in datasets:
    # create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(images), 48, 48), outputPath)

    for (image, label) in zip(images, labels):
        writer.add([image], [label])

    writer.close()

f.close()