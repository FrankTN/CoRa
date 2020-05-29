import csv

import radiomics
from radiomics import generalinfo
import utilities as ut
import radiomics_funcs as rf

HI_VERBOSITY = 10
LO_VERBOSITY = 40

if __name__ == '__main__':
    radiomics.setVerbosity(HI_VERBOSITY)
    rf.setup_logger()

    f_extractor = rf.initialize_extractor('params.yaml')
    file_list = ut.read_files('data_paths.csv')

    # image = sitk.ReadImage(file_list[0][0])
    # mask = sitk.ReadImage('sampled.nii')

    info = generalinfo.GeneralInfo()
    print(info.getGeneralInfo())

    features = rf.extract_features(file_list, f_extractor)

    ut.print_features(features)
    ut.store_features(features)
