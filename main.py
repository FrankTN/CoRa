import os
from multiprocessing import cpu_count

import radiomics_funcs as rf
import utilities as ut

# Setup structured filepaths
ROOT = os.getcwd()
PARAMS = os.path.join(ROOT, 'params.yaml')
LOG = os.path.join(ROOT, 'log.txt')
INPUT_CSV = os.path.join(ROOT, 'cases.csv')
OUTPUT_CSV = os.path.join(ROOT, 'results.csv')

if __name__ == '__main__':

    # Write logs to logfile, set verbosity
    lgr = rf.setup_logger(LOG)

    # Setting for extractor are defined in params.yaml, files to process are defined in cases.csv
    f_extractor = rf.initialize_extractor(PARAMS, lgr)
    file_list = ut.read_files(INPUT_CSV, lgr)

    # Perform the feature calculation and return vector of features
    features = rf.extract_features(file_list, f_extractor)

    # Currently we print the results to the screen and we store them in results.csv
    ut.print_features(features)
    ut.store_features(features, file_list, OUTPUT_CSV, lgr)
