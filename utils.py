from typing import Generator
import pandas as pd


def read_colvar(filename: str) -> pd.DataFrame:
    with open(filename, "r") as read_file:
        line = read_file.readline()
        main_line = line.strip().split()[2::]
        nskips = 0
        while line.startswith("#"):
            nskips += 1
            line = read_file.readline()

    data = pd.read_csv(filename, sep="\s+", names=main_line, skiprows=nskips)
    return data


def line_generator(file) -> Generator:
    while True:
        line = file.readline()
        if line:
            yield line
        else:
            break
