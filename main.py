import radiomics_funcs as rf
import utilities as ut

if __name__ == '__main__':
    # Write logs to testLog.txt, set verbosity
    rf.setup_logger()

    # Setting for extractor are defined in params.yaml, files to process are defined in data_paths.csv
    f_extractor = rf.initialize_extractor('params.yaml')
    file_list = ut.read_files('data_paths.csv')

    # Perform the feature calculation and return vector of features
    features = rf.extract_features(file_list, f_extractor)

    # Currently we print the results to the screen and we store them in results.csv
    ut.print_features(features)
    ut.store_features(features, file_list, 'results.csv')
