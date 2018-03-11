# IMPORTS.
import os               # path finding.
import cv2 as cv        # webcam image.
import time             # sleep time.
import numpy as np      # array calculations

def train(recognizer, training_set_path):

    training_time = time.time()

    # Prepare training set for training

    images = []
    labels = []
    label_ids = []

    label_id = 0

    prep_time = time.time()

    subdirs = os.listdir(training_set_path)

    for dir_name in subdirs:

        # Skip system files
        if dir_name.startswith('.'):
            continue

        # Get label from folder name
        label = dir_name.split('.')[0]

        # Save it in label list
        labels.append(label)

        # Suild subdir path
        subdir = training_set_path + '/' + dir_name

        # Get list of files
        files = os.listdir(subdir)

        for file_name in files:

            # Skip system files
            if file_name.startswith('.'):
                continue

            # Get file path
            file_path = subdir + '/' + file_name

            # Read the image
            image = cv.imread(file_path, 0)

            # Build images and label_ids list
            images.append(image)
            label_ids.append(label_id)

        # Increase label id iterator
        label_id += 1

    prep_time = time.time() - prep_time

    print('End preparation of training set after ' + str(round(prep_time,2)) + 's.')
    print('Number of images: ' + str(len(images)))
    print('Number of trained labels: ' + str(len(labels)))



    print('Start training...')

    # Train the recognizer
    recognizer.train(images, np.array(label_ids))

    label_id = 0

    for label in labels:
        recognizer.setLabelInfo(label_id, label)

        label_id += 1

    training_time = time.time() - training_time
    print('End training after ' + str(round(training_time,2)) + 's.')

def capture(detector, label, classifier = "lbp", resize = False, resize_factor = 0.5, make_training_set = True, training_set_size = 20, make_validation_set = False, validation_set_size = 20, training_set_path = 'data/training', validation_set_path = 'data/validation'):

    # Get camera image.
    camera = cv.VideoCapture(0)
    output, camera_image = camera.read()

    # Initialize iterator.
    iterator = 0

    # Set number of images.
    no_of_images = 0
    if make_validation_set is True:
        no_of_images = training_set_size + validation_set_size
    else:
        no_of_images = training_set_size

    while output and iterator < no_of_images:

        path = None

        # Change path depending on validation and training set mode.
        if iterator < training_set_size:
            path = training_set_path
        elif make_validation_set:
            path = validation_set_path

        # Capture face images.
        faces, eyes, image = detector.detect(image = camera_image, classifier = classifier, resize = resize, resize_factor = resize_factor, detect_eyes = False, draw_bounding_boxes = False, save_faces = True, face_path = path, filename = label)

        # Change mode to validation. User has to hit enter to be prepared and vary images.
        if iterator is training_set_size and make_validation_set:
            input("Now the computer will save some variations of your face. Please tilt and rotate your face a bit, laugh or grin, make some grimaces in front of the camera to enhance the recognition even in unusual states. Hit enter when you are ready.\n")

        # Show results.
        cv.imshow('Face capturing', image)

        # Wait for the next image for more variability.
        time.sleep(0.1)

        if len(faces) > 0:
            iterator += 1

        # Grab next camera image
        output, camera_image = camera.read()

    cv.destroyAllWindows()
    camera.release()
