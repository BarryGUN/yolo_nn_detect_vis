import asyncio
import os.path
import subprocess
import sys
from asyncio.queues import Queue
from pathlib import Path


def get_variable_from_file(filepath, variable_name):
    variables = {}
    filepath = Path(filepath)
    with open(filepath, 'r') as file:
        code = file.read()
        exec(code, {}, variables)

    return variables.get(variable_name), filepath.stem, filepath.parent


def dict_to_command_params(params):
    """
    Convert a dictionary of parameters into a list of command line arguments.

    :param params: A dictionary where keys are parameter names and values are parameter values.
    :return: A list of strings representing the command line arguments.
    """
    command_params = []
    for key, value in params.items():
        # Add the key as an option flag (e.g., --key or -k for single character keys)
        if len(key) == 1:
            command_params.append(f'-{key}')
        else:
            command_params.append(f'--{key}')

        # Add the value for the option flag
        if isinstance(value, bool):
            # If the value is a boolean, only add it if it's True
            if value:
                continue
        elif isinstance(value, list):
            # If the value is a list, join the items with spaces
            command_params.append(' '.join(map(str, value)))
        elif value == '':
            command_params.append('\'\'')
        else:
            # For other types, add the value as a string
            command_params.append(str(value))

    return command_params


def command_generater(exc_file, config_file, save=False):
    param, name, project = get_variable_from_file(config_file, variable_name='config')
    param = dict_to_command_params(param)
    param = ' '.join(map(str, param))
    cmd = f'python {exc_file} {param}'
    if save:
        project = f'{project}{os.sep}cmd'
        if not os.path.exists(project):
            os.makedirs(project)
        save_path = f'{project}{os.sep}{name}'
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(cmd)
        print(f'cmd saved in {save_path}')

    return cmd


if __name__ == '__main__':
    command_generater(exc_file='train.py',
                      config_file='configs/train/yolonn-n-expfree_ciou_640-custom.py',
                      save=True)
