import csv
import os

import med2image


def print_features(features_list) -> None:
    """Prints the feature name and its value"""
    for features in features_list:
        for featureName in features.keys():
            print("Computed %s: %s" % (featureName, features[featureName]))


def read_files(file_path, logger):
    """ Reads a csv file containing pairs of scan names and masks, returns a list of masks """
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile, quotechar='|')
            # Skips the header in the file
            next(reader)
            path_list = []
            for row in reader:
                data_path = row[0]
                mask_path = row[1]
                label = row[2]
                path_list.append((data_path, mask_path, label))
    except IOError:
        logger.error('Unable to read file {}'.format(file_path), exc_info=True)
    logger.info('Input files loaded')
    return path_list


def store_features(features, file_names, out_path, logger):
    # Store the calculated features in a csv file in default pyradiomics batch output style
    if not features:
        return
    try:
        # Take the parameter names from the first feature vector
        csv_columns = ['Image', 'Mask', *list(features[0].keys())]
        with open(out_path, 'w') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=csv_columns)
            writer.writeheader()
            for scan, file_name in zip(features, file_names):
                scan['Image'] = file_name[0]
                scan['Mask'] = file_name[1]
                writer.writerow(scan)
    except IOError:
        logger.error('Unable to write to output: {}'.format(out_file), exc_info=True)
    logger.info('Done writing to file: ' + out_path)


def create_input_names(out_path, case_type):
    """ Populates the input path csv at out_path, using hardcoded filenames"""
    csv_columns = ['Image', 'Mask', 'Label']
    with open(out_path, 'w') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=csv_columns)
        writer.writeheader()
        case_type(writer)


def write_mosmed(writer, sampled: bool = True):
    """Writes the cases for the Moscow dataset to the writer object."""
    pair = {}
    for i in range(255, 305):
        if sampled:
            pair['Image'] = "data/COVID19_1110/studies/CT-1/study_0" + str(i) + ".nii"
            pair['Mask'] = "data/COVID19_1110/masks/study_0" + str(i) + "_mask_sampled_1.nii"
            writer.writerow(pair)
        else:
            pair['Image'] = "data/COVID19_1110/studies/CT-1/study_0" + str(i) + ".nii"
            pair['Mask'] = "data/COVID19_1110/masks/study_0" + str(i) + "_mask.nii"
            writer.writerow(pair)


def write_medseg(writer, target=10, sampled: bool = False):
    """Writes the cases for the Italian dataset to the writer object."""
    pair = {}
    for i in range(1, target):
        if sampled:
            pair['Image'] = "data/rp_im/" + str(i) + ".nii"
            pair['Mask'] = "data/rp_msk/" + str(i) + "_sampled_1.nii"
            pair['Label'] = "1"
            writer.writerow(pair)
            pair['Image'] = "data/rp_im/" + str(i) + ".nii"
            pair['Mask'] = "data/rp_msk/" + str(i) + "_sampled_2.nii"
            pair['Label'] = "2"
            writer.writerow(pair)
        else:
            pair['Image'] = "data/rp_im/" + str(i) + ".nii"
            pair['Mask'] = "data/rp_msk/" + str(i) + ".nii"
            writer.writerow(pair)


def write_simple(writer):
    """Writes the first 2 cases of the Italian dataset to the writer object."""
    write_medseg(writer, 2)


def convert_nifti_to_png(file_list):
    if all(file[0].endswith('.nii') and file[1].endswith('.nii') for file in file_list):
        for file in file_list:
            cmd = 'med2image -i ' + file[0] + ' -d ' + file[0].split('.')[0] + ' -t png'
            os.system(cmd)
            cmd = 'med2image -i ' + file[1] + ' -d ' + file[1].split('.')[0] + '_msk' + ' -t png'
            os.system(cmd)

    else:
        print("Error, not all files in list are in nifti format")
