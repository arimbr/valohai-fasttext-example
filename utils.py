import os


def get_first_file(path):
    filename = os.listdir(path)[0]
    return os.path.join(path, filename)


def get_input_path(input_name):
    '''
    Args:
        input_name (str): the input name in the valohai.yaml.
        Locally pass the relative path to the input file.
    '''
    inputs_dir = os.getenv('VH_INPUTS_DIR')
    if inputs_dir:
        input_dir = os.path.join(inputs_dir, input_name)
        return get_first_file(input_dir)
    return input_name


def get_output_path(output_file):
    '''
    Args:
        output_file (str): the output file name.
        Locally pass the relative path to the output file.
    '''
    outputs_dir = os.getenv('VH_OUTPUTS_DIR')
    if outputs_dir:
        return os.path.join(outputs_dir, output_file)
    return output_file
