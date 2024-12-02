import argparse

parser = argparse.ArgumentParser(description="params")
# parser.add_argument('--nargs', type=int, nargs="+", default=[256, 256])
parser.add_argument('--nargs2', type=int, nargs="?", help="input zero or one param. The former denotes None.")
args = parser.parse_args()

# To show the results of the given option to screen.
for _, value in parser.parse_args()._get_kwargs():
    print(f"value: {value}, type: {type(value)}")
    if value is not None:
        print(f"not none value: {value}, type: {type(value)}")
