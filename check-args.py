import argparse

parser = argparse.ArgumentParser(description="params")
parser.add_argument('--nargs', type=int, nargs="+", default=[256, 256])
args = parser.parse_args()

# To show the results of the given option to screen.
for _, value in parser.parse_args()._get_kwargs():
    if value is not None:
        print(value)
