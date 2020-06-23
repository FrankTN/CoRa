import os
import multiprocessing as mp
import click

import radiomics_funcs as rf
import utilities as ut

# Setup structured filepaths
ROOT = os.getcwd()
PARAMS = os.path.join(ROOT, 'params.yaml')
LOG = os.path.join(ROOT, 'log.txt')
INPUT_CSV = os.path.join(ROOT, 'cases.csv')
OUTPUT_CSV = os.path.join(ROOT, 'results_d.csv')

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
def run(input_f, output_f, params, log, parallel):
    # Write logs to logfile, set verbosity
    lgr = rf.setup_logger(log)

    # Setting for extractor are defined in params.yaml, files to process are defined in cases.csv
    f_extractor = rf.initialize_extractor(params, lgr)
    file_list = ut.read_files(input_f, lgr)

    if parallel:
        click.echo("Parallel mode enabled")
        click.echo("Number of processors: {}".format(mp.cpu_count()))
        pool = mp.Pool(CPU_COUNT)

        # Perform the feature calculation and return vector of features
        result_objects = [pool.apply_async(rf.extract_features, args=(file, f_extractor)) for file in
                          file_list]

        # Unpack the worker results back into desired features
        features = [r.get() for r in result_objects]

        # Cleanup after parallel work
        pool.close()
        pool.join()
    else:
        features = list()
        for file in file_list:
            features.append(rf.extract_features(file, f_extractor))

    # Currently we print the results to the screen and we store them in results.csv
    ut.store_features(features, file_list, output_f, lgr)


# @cora.command()
# @click.option('-p', '--set-pars', type=click.Choice(['simple', 'full'], case_sensitive=False))
# def params(set_pars):
#     click.echo(set_pars)


@cora.command()
@click.option('-o', '--output-f', default=INPUT_CSV, help='Cases target file')
@click.option('-c', '--case-type', type=click.Choice(['medseg', 'mosmed'], case_sensitive=False), help="define which "
                                                                                                       "dataset to "
                                                                                                       "prepare")
def cases(output_f, case_type):
    """ Creates a case file .csv based on the type of dataset being analyzed"""
    if case_type == 'medseg':
        ut.create_input_names(output_f, ut.write_medseg)
    elif case_type == 'mosmed':
        pass


@cora.command()
@click.confirmation_option(prompt='Are you sure you want to remove all .csv files?')
def clean():
    click.echo("Removing all .csv files")
    files_in_directory = os.listdir(os.getcwd())
    filtered_files = [file for file in files_in_directory if file.endswith(".csv")]
    for file in filtered_files:
        path_to_file = os.path.join(os.getcwd(), file)
        os.remove(path_to_file)


def main():
    cora(prog_name='cora')


if __name__ == '__main__':
    main()
