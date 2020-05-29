import csv


def print_features(features_list) -> None:
    for features in features_list:
        for featureName in features.keys():
            print("Computed %s: %s" % (featureName, features[featureName]))


def read_files(file_path):
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
        print("I/O error")
    return path_list


def store_features(features, target_file):
    # Store the calculated features in a csv file
    csv_columns = ['Name', 'Value']

    try:
        with open(target_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_columns)
            for scan in features:
                for key, value in scan.items():
                    writer.writerow([key, value])
    except IOError:
        print("I/O error")
