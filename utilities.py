import csv


def print_features(features_list) -> None:
    for features in features_list:
        for featureName in features.keys():
            print("Computed %s: %s" % (featureName, features[featureName]))


def read_files(file_path, logger):
    """ Reads a csv file containing pairs of scans and masks """
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile, quotechar='|')
            # Skips the header in the file
            next(reader)
            path_list = []
            for row in reader:
                data_path = row[0]
                mask_path = row[1]
                path_list.append((data_path, mask_path))
    except IOError:
        logger.error('Unable to read file {}'.format(file_path), exc_info=True)
    logger.info('Input files loaded')
    return path_list


def store_features(features, file_names, out_path, logger):
    # Store the calculated features in a csv file in default pyradiomics batch output style
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
