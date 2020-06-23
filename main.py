import os
import multiprocessing as mp

import radiomics_funcs as rf
import utilities as ut

# Setup structured filepaths
ROOT = os.getcwd()
PARAMS = os.path.join(ROOT, 'params.yaml')
LOG = os.path.join(ROOT, 'log.txt')
INPUT_CSV = os.path.join(ROOT, 'cases.csv')
OUTPUT_CSV = os.path.join(ROOT, 'results_subsampled_3_1.csv')

# Setup other constants
CPU_COUNT = mp.cpu_count()

if __name__ == '__main__':
    # ut.create_input_names(INPUT_CSV)

    print("Number of processors: {}".format(mp.cpu_count()))
    pool = mp.Pool(CPU_COUNT)

    # Write logs to logfile, set verbosity
    lgr = rf.setup_logger(LOG)

    # Setting for extractor are defined in params.yaml, files to process are defined in cases.csv
    f_extractor = rf.initialize_extractor(PARAMS, lgr)
    file_list = ut.read_files(INPUT_CSV, lgr)

    # rf.sample_masks(file_list)

    # Perform the feature calculation and return vector of features
    result_objects = [pool.apply_async(rf.extract_features, args=(file, f_extractor)) for file in
                      file_list]

    # Unpack the worker results back into desired features
    features = [r.get() for r in result_objects]

    # Cleanup after parallel work
    pool.close()
    pool.join()

    # Currently we print the results to the screen and we store them in results.csv
    ut.store_features(features, file_list, OUTPUT_CSV, lgr)
