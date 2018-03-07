#######################################################
# Face trainer - face extractor application
# @author: Christian Reichel
# Version: 0.1a
# -----------------------------------------------------
# This script makes a directory based on an entered
# name and a hash value and saves X faces into that
# folder as training set for face recognition.
#######################################################

# IMPORTS
from modules.face_extractor import face_extractor as extractor
from modules.face_detector import face_detector as detector

extractor.build_training_set(detector)