# CV module: Face Trainer

Trains a given OpenCV recognition object by preparing the given training set and call the `train` function of the recognizer.

The training set directories have to have the format `LABEL.IGNORED_INFORMATION` where the label has to be before the dot `.` in the directory name.

A suitable folder structure might be the following:

```
.
.
├── training_set
|   ├── label.1
|   |   ├── image.1.jpg
|   |   ├── . . .
|   |   └── image.n.jpg
|   ├── label.2
|   ├──  . . .
|   └── label.n
.
.
```

## Installation and usage

1. Open a terminal window, navigate to your project and add this python as submodule as following:

```bash
git submodule add https://github.com/n1g1r1/cv-module-face-trainer modules/face_trainer
```

2. Import it as python module:

```python
from modules.face_trainer import trainer
```

3. Call the train function:

```python
train(recognizer, training_set_path)
```

#### Parameters

- `recognizer`: The given OpenCV recognizer.
- `training_set_path`: The path to the training set.

### Capture functon

Or if you have to capture some webcam images first:

```python
capture(detector, label, classifier = "lbp", resize = False, resize_factor = 0.5, make_training_set = True, training_set_size = 20, make_validation_set = False, validation_set_size = 20, training_set_path = 'data/training', validation_set_path = 'data/validation')
```

#### Parameters

- `detector`: The detector module to use it for the face detection.
- `label`: The label that has to be used to make the set directory and save the images with the correct naming.
- `classifier`: The face classifier method that will be used for the face detection. Default: `lbp`.
- `resize`: Should the image get resized? Default: `False`.
- `resize_factor`: Resize factor, if resized. Default: `0.5`.
- `make_training_set`: Should the algorithm make the training set? Default: `True`.
- `training_set_size`: How many images will be shot and saved. Default: `20`.
- `make_validation_set`: Should the algorithm make the validation set? Default: `False`.
- `validation_set_size`: How many images will be shot and saved. Default: `20`
- `training_set_path`: Where the training set gets stored. Default: `data/training`.
- `validation_set_path`: Where the validation set gets stored. Default: `data/validation`.
