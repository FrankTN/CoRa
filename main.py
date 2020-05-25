import logging

import radiomics
from radiomics import featureextractor
import SimpleITK as sitk
import os


def extract_features(names, extractor):
    """ This function extracts features from a list of file and masks as specified by the names parameter
        returns a feature vector
    """
    featureVector = list()
    print("Calculating features")
    for (imageName, maskName) in names:
        featureVector.append(extractor.execute(imageName, maskName))
    return featureVector


def print_features(features_list):
    for features in features_list:
        for featureName in features.keys():
            print("Computed %s: %s" % (featureName, features[featureName]))


# get pyradiomics logger, loglevel DEBUG
logger = radiomics.logger
logger.setLevel(logging.DEBUG)

# Set up the handler to write out all log entries to a file
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Define settings for signature calculation
# These are currently set equal to the respective default values
settings = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkBSpline}
print(settings)

# Initialize feature extractor
f_extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

# Disable all classes except firstorder
f_extractor.disableAllFeatures()

# Only enable mean and skewness in firstorder
f_extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])

imageName = os.path.realpath('data/tr_im.nii')
maskName = os.path.realpath('data/tr_mask.nii')

filenames = list()
filenames.append((imageName, maskName))
features = extract_features(filenames, f_extractor)
print_features(features)
