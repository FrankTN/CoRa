import logging
import os

import SimpleITK as sitk
import numpy as np
import radiomics
from radiomics import featureextractor, generalinfo
from scipy import ndimage
from random import randint

HI_VERBOSITY = 10
LO_VERBOSITY = 40


def setup_logger():
    radiomics.setVerbosity(HI_VERBOSITY)
    # get pyradiomics logger, loglevel DEBUG
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)

    # Set up the handler to write out all log entries to a file
    handler = logging.FileHandler(filename='testLog.txt', mode='w')
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def ROI_sampling(mask: sitk.Image) -> sitk.Image:
    """
    Clips the default mask using a standard area, overcoming the area dependence of certain first order measures such as
    Shannon entropy.
    """
    # The original type is float64, we convert to int32
    discretized_mask = sitk.Cast(mask, sitk.sitkInt32)
    output = sitk.Image(mask.GetSize(), sitk.sitkInt32)

    # For each slice in the mask we try to find a suitable region
    for z in np.arange(discretized_mask.GetDepth()):
        # Extract a slice from the mask
        img = discretized_mask[:, :, z.item()]
        img_arr = sitk.GetArrayFromImage(img)

        # Erode the mask using a 20 x 20 box
        img_arr_eroded = ndimage.binary_erosion(img_arr, structure=np.ones((20, 20)))
        out_arr = sitk.GetArrayFromImage(output[:, :, z.item()])

        # Pick a random location within the eroded mask, this will be the new center of our window
        indices = np.nonzero(img_arr_eroded)
        # Check if there are any indices in the tuple
        if len(indices[0]):
            random_index = randint(0, len(indices[0]) - 1)
            # Create a new image of the intersect of the mask with the selected window
            c_x = indices[0][random_index]
            c_y = indices[1][random_index]

            out_arr[c_x - 10:c_x + 10, c_y - 10:c_y + 10] = img_arr[c_x - 10:c_x + 10, c_y - 10:c_y + 10]
            assert np.count_nonzero(out_arr[c_x - 10:c_x + 10, c_y - 10:c_y + 10]) == 400
        # Paste the new image into the output image
        img_vol = sitk.JoinSeries(sitk.GetImageFromArray(out_arr))
        output = sitk.Paste(output, img_vol, img_vol.GetSize(), destinationIndex=[0, 0, z.item()])
    output.CopyInformation(mask)
    return output


def initialize_extractor(parameters: str, logger: radiomics.logger) -> featureextractor.RadiomicsFeatureExtractor:
    # Initialize feature extractor, if inputfile is valid
    if os.path.isfile(parameters):
        extractor = featureextractor.RadiomicsFeatureExtractor(parameters)
    else:  # Parameter file not found, use hardcoded settings instead
        settings = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkBSpline,
                    'enableCExtensions': True}
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    logger.info("Parameters loaded")
    return extractor


def extract_features(files: list, extractor: featureextractor.RadiomicsFeatureExtractor):
    """
    This function extracts features from a list of file and masks as specified by the names parameter
    returns a feature vector
    """
    feature_vector = list()
    print("Calculating features")
    # TODO Efficiently extract for all labels in mask
    for (image, mask) in files:
        feature_vector.append(extractor.execute(image, mask))
    return feature_vector


def print_img_info(image: sitk.Image) -> None:
    print("Pixel Type    {}".format(image.GetPixelID()))
    print("Size          {}".format(image.GetSize()))
    print("Origin        {}".format(image.GetOrigin()))
    print("Spacing       {}".format(image.GetSpacing()))
    print("Direction     {}".format(image.GetDirection()))


def print_gen_info() -> None:
    info = generalinfo.GeneralInfo()
    print(info.getGeneralInfo())
