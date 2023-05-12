# filename: count_lines.py

import json
import argparse


def count_lines(notebook_path):
    with open(notebook_path) as f:
        data = json.load(f)

    code_lines = 0
    documentation_lines = 0

    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            code_lines += sum(len(line.strip()) > 0 for line in cell['source'])
        elif cell['cell_type'] == 'markdown':
            documentation_lines += sum(len(line.strip())
                                       > 0 for line in cell['source'])

    print(f'Lines of code: {code_lines}')
    print(f'Lines of documentation: {documentation_lines}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Count lines of code and documentation in a Jupyter notebook.')
    parser.add_argument('notebook_path', type=str,
                        help='Path to the Jupyter notebook file.')
    args = parser.parse_args()

    count_lines(args.notebook_path)
