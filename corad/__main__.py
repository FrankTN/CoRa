import os
import multiprocessing as mp
import click
from tqdm import tqdm

import radiomics_funcs as rf
import utilities as ut

# Setup structured filepaths
ROOT = os.getcwd()
PARAMS = os.path.join(ROOT, 'params.yaml')
LOG = os.path.join(ROOT, 'log.txt')
INPUT_CSV = os.path.join(ROOT, 'cases.csv')
OUTPUT_CSV = os.path.join(ROOT, 'results.csv')

# Setup other constants
CPU_COUNT = mp.cpu_count()


@click.group()
def cora():
    """A CLI wrapper for Cora, COVID Radiomics"""


@cora.command()
@click.option('-i', '--input-f', default=INPUT_CSV, help='Input file, containing list of files and corresponding '
                                                         'masks, a third label column is optional')
@click.option('-o', '--output-f', default=OUTPUT_CSV, help='Output target file')
@click.option('-p', '--params', default=PARAMS, help='Parameter file, default params.yaml')
@click.option('-l', '--log', default=LOG, help='Log location, default log.txt')
@click.option('-p', '--parallel', default=False, type=bool, is_flag=True, help='Parallelization flag')
@click.option('-b', '--label', default=1, type=int, help='The label to be used in the extraction, has to be valid for '
                                                         'all masks being used. Note that any label defined in the '
                                                         'input file takes precedence.')
def run(input_f, output_f, params, log, parallel, label):
    extract(input_f, output_f, params, log, parallel, label)


def extract(input_f, output_f, params, log, parallel, label):
    # Write logs to logfile, set verbosity
    lgr = rf.setup_logger(log)

    # Setting for extractor are defined in params.yaml, files to process are defined in cases.csv
    f_extractor = rf.initialize_extractor(params, lgr)
    file_list = ut.read_files(input_f, lgr)

    if parallel:
        click.echo("Parallel mode enabled")
        click.echo("Number of processors: {}".format(mp.cpu_count()))
        click.echo("Extracting features")
        pool = mp.Pool(CPU_COUNT)
        prog_bar = tqdm(total=len(file_list))

        # Perform the feature calculation and return vector of features
        # TODO no logger, cant pickle
        result_objects = [pool.apply_async(rf.extract_features, args=(file, f_extractor, output_f, label),
                                           callback=lambda _: prog_bar.update(1)) for file in file_list]
        # Unpack the worker results back into desired features
        features = [r.get() for r in result_objects]
        features = [i for i in features if i]

        # Cleanup after parallel work
        pool.close()
        pool.join()
    else:
        features = list()
        click.echo("Extracting features")
        for file in tqdm(file_list):
            result = rf.extract_features(file, f_extractor, output_f, label, lgr)
            if result:
                features.append(result)

    # Currently we print the results to the screen and we store them in results.csv
    # ut.store_features(features, file_list, output_f, lgr)


# @cora.command()
# @click.option('-p', '--set-pars', type=click.Choice(['simple', 'full'], case_sensitive=False))
# def params(set_pars):
#     click.echo(set_pars)

@cora.command()
def test():
    """ Runs a test case using simple parameters"""
    ut.create_input_names(INPUT_CSV, ut.write_simple)
    extract(INPUT_CSV, OUTPUT_CSV, PARAMS, LOG, False)


@cora.command()
def masks():
    target = os.path.join(os.getcwd(), 'data/UMCG/DENOISED')
    ut.create_masks(target)


@cora.command()
@click.option('-o', '--output-f', default=INPUT_CSV, help='Cases target file')
@click.option('-c', '--case-type', type=click.Choice(['medseg', 'mosmed', 'simple', 'UMCG_R', 'UMCG_D'], case_sensitive=False), help=
"Define which dataset to prepare")
@click.option('-s', '--sampled', is_flag=True, help="If set will write subsampled cases")
def cases(output_f, case_type, sampled):
    """ Creates a case file .csv based on the type of dataset being analyzed"""
    if case_type == 'medseg':
        ut.create_input_names(output_f, ut.write_medseg, sampled)
    elif case_type == 'simple':
        ut.create_input_names(output_f, ut.write_simple)
    elif case_type == 'mosmed':
        ut.create_input_names(output_f, ut.write_mosmed, sampled)
    elif case_type == 'UMCG_R':
        ut.create_input_names(output_f, ut.write_UMCG, sampled)
    elif case_type == 'UMCG_D':
        ut.create_input_names(output_f, ut.write_UMCG_D, sampled)

@cora.command()
@click.confirmation_option(prompt='Are you sure you want to remove all .csv files?')
def clean():
    click.echo("Removing all .csv files")
    files_in_directory = os.listdir(os.getcwd())
    filtered_files = [file for file in files_in_directory if file.endswith(".csv")]
    for file in filtered_files:
        path_to_file = os.path.join(os.getcwd(), file)
        os.remove(path_to_file)


@cora.command()
@click.option('-i', '--input-f', default=INPUT_CSV, help='Input file, containing list of files and corresponding '
                                                         'masks, a third label column is optional')
def sample(input_f):
    lgr = rf.setup_logger(LOG)
    file_list = ut.read_files(input_f, lgr)
    rf.sample_masks(file_list)


@cora.command()
def convert():
    """ This command is used to prepare for CNN processing, by converting everything to png files."""
    lgr = rf.setup_logger('log.txt')
    ut.convert_nifti_to_png(ut.read_files(INPUT_CSV, lgr))


def main():
    cora(prog_name='cora')


if __name__ == '__main__':
    main()
