import logging
from random import randint

import numpy as np
from scipy import ndimage

import radiomics
from radiomics import featureextractor
from radiomics import generalinfo

import SimpleITK as sitk
import os


# import utilities


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

            out_arr[c_x-10:c_x+10, c_y-10:c_y+10] = img_arr[c_x-10:c_x+10, c_y-10:c_y+10]
            assert np.count_nonzero(out_arr[c_x-10:c_x+10, c_y-10:c_y+10]) == 400
        # Paste the new image into the output image
        img_vol = sitk.JoinSeries(sitk.GetImageFromArray(out_arr))
        print(img_vol.GetOrigin())
        output = sitk.Paste(output, img_vol, img_vol.GetSize(), destinationIndex=[0, 0, z.item()])
    output.CopyInformation(mask)
    return output


def extract_features(files, extractor):
    """
    This function extracts features from a list of file and masks as specified by the names parameter
    returns a feature vector
    """
    featureVector = list()
    print("Calculating features")
    for (image, mask) in files:
        featureVector.append(extractor.execute(image, mask))
    return featureVector


def print_features(features_list):
    for features in features_list:
        for featureName in features.keys():
            print("Computed %s: %s" % (featureName, features[featureName]))


# Currently uses hardcoded paths
imageName = os.path.realpath('data/tr_im.nii')
maskName = os.path.realpath('data/tr_mask.nii')

image = sitk.ReadImage(imageName)
mask = sitk.ReadImage(maskName)

sitk.WriteImage(ROI_sampling(mask), 'sampled.nii')
