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


def setup_logger(log_path):
    radiomics.setVerbosity(HI_VERBOSITY)
    # get pyradiomics logger, loglevel DEBUG
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)

    # Set up the handler to write out all log entries to a file
    handler = logging.FileHandler(filename=log_path, mode='w')
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
        # Paste the new image into the output image
        img_vol = sitk.JoinSeries(sitk.GetImageFromArray(out_arr))
        output = sitk.Paste(output, img_vol, img_vol.GetSize(), destinationIndex=[0, 0, z.item()])
    output.CopyInformation(mask)
    return output


def initialize_extractor(parameters: str, logger: radiomics.logger) -> featureextractor.RadiomicsFeatureExtractor:
    # Initialize feature extractor, if inputfile is valid
    if os.path.isfile(parameters):
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(parameters)
    else:  # Parameter file not found, use hardcoded settings instead
        settings = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkBSpline,
                    'enableCExtensions': True}
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(**settings)
    logger.info("Parameters loaded")
    return extractor


def extract_features(files: list, extractor: radiomics.featureextractor.RadiomicsFeatureExtractor,
                     logger: radiomics.logger):
    """
    Reads a tuple of file and mask, extracts features
    """
    image, mask, label = files
    print("Calculating features")
    # TODO Efficiently extract for all labels in mask
    print(image, mask)
    lab_val = None
    if label:
        lab_val = int(label)
    result = None
    try:
        result = extractor.execute(image, mask, label=lab_val)
    except ValueError:
        logger.warning("Unable to extract features from image {}")
    return result


def print_img_info(image: sitk.Image) -> None:
    print("Pixel Type    {}".format(image.GetPixelID()))
    print("Size          {}".format(image.GetSize()))
    print("Origin        {}".format(image.GetOrigin()))
    print("Spacing       {}".format(image.GetSpacing()))
    print("Direction     {}".format(image.GetDirection()))


def print_gen_info() -> None:
    info = generalinfo.GeneralInfo()
    print(info.getGeneralInfo())


def sample_masks(file_list):
    for (_, mask_name) in file_list:
        mask = sitk.Cast(sitk.ReadImage(mask_name), sitk.sitkInt32)

        # Hardcoded the levels right now, these correspond to the labels within the masks
        low = 1
        high = 2
        for lvl in np.arange(low, high + 1):
            tmp_mask = sitk.GetArrayFromImage(mask)
            tmp_mask[tmp_mask != lvl] = 0
            tmp_mask_img = sitk.GetImageFromArray(tmp_mask)
            tmp_mask_img.CopyInformation(mask)
            updated_mask = ROI_sampling(tmp_mask_img)
            mask_prefix, mask_extension = os.path.splitext(mask_name)
            sitk.WriteImage(updated_mask, mask_prefix + "_sampled_" + str(lvl) + mask_extension)