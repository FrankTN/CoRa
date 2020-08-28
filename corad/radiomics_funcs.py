import csv
import logging
import os

import SimpleITK as sitk
import numpy as np
import radiomics
from radiomics import featureextractor
from filelock import FileLock

from scipy import ndimage
from random import randint

'''
    This file contains functions which interact with the Pyradiomics library.
    Frank te Nijenhuis 2020
'''

# These constants are used by the logger to switch verbosity levels.
HI_VERBOSITY = 10
LO_VERBOSITY = 40


def setup_logger(log_path):
    radiomics.setVerbosity(LO_VERBOSITY)
    # Get pyradiomics logger, loglevel DEBUG
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
    discretized_mask = sitk.Cast(mask, sitk.sitkUInt32)
    output = sitk.Image(mask.GetSize(), sitk.sitkUInt32)

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
    else:
        logger.warning('Parameter file not found, use hardcoded settings instead')
        settings = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkBSpline,
                    'enableCExtensions': True}
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(**settings)
    logger.info("Parameters loaded")
    return extractor


def extract_features(files: list, extractor: radiomics.featureextractor.RadiomicsFeatureExtractor, output_csv,
                      lab_val: int = 1, logger: radiomics.logger = None):
    """
    Reads a tuple of file and mask, extracts features
    """

    # Do this to handle parallel processing where we can't pass the logger
    if not logger:
        info = warning = print
    else:
        info = logger.info
        warning = logger.warning

    image, mask, label = files
    # TODO Efficiently extract for all labels in mask
    if label:
        # Label defined in the input file takes precedence over the argument
        info('Overriding manual label (-b) parameter, was ' + str(lab_val) + ', now ' + label)
        lab_val = int(label)
    try:
        result = extractor.execute(image, mask, label=lab_val)
        # write to file

    except ValueError as err:
        warning("Unable to extract features, error: {}".format(err))
        return None

    store_row(image, mask, result, output_csv, logger)
    # info('Extraction successful: \t' + image + '\t' + mask)
    return result


def store_row(img, msk, features, out_path, logger):
    # Store the calculated features in a csv file in default pyradiomics batch output style
    if not features:
        logger.warning('Can\'t store output, no features to store, continuing')
        return
    try:
        with FileLock(out_path + '.lock'):
            out_file = open(out_path, 'a')
            csv_columns = ["Image", "Mask", *list(features.keys())]
            writer = csv.DictWriter(out_file, fieldnames=csv_columns)
            if os.path.getsize(os.path.join(os.getcwd(), out_path)) == 0:
                # File is empty, we can write the header
                writer.writeheader()
            features['Image'] = img
            features['Mask'] = msk
            writer.writerow(features)
            out_file.flush()
    except ValueError as err:
        print(err)


def sample_masks(file_list):
    for (_, mask_name, _) in file_list:
        mask = sitk.Cast(sitk.ReadImage(mask_name), sitk.sitkInt32)

        # Hardcoded the levels right now, these correspond to the labels within the masks
        low = 1
        high = 5
        for lvl in np.arange(low, high + 1):
            tmp_mask = sitk.GetArrayFromImage(mask)
            tmp_mask[tmp_mask != lvl] = 0
            tmp_mask_img = sitk.GetImageFromArray(tmp_mask)
            tmp_mask_img.CopyInformation(mask)
            updated_mask = ROI_sampling(tmp_mask_img)
            mask_prefix, mask_extension = os.path.splitext(mask_name)
            sitk.WriteImage(updated_mask, mask_prefix + "_sampled_" + str(lvl) + mask_extension)