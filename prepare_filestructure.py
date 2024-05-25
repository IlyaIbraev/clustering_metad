import os


def create_directory(directory: str, path: str) -> None:
    if directory not in os.listdir(path):
        os.mkdir(
            os.path.join(path, directory)
        )


def prepare_main_calc_directory() -> None:
    create_directory("calculations", "")
    create_directory("trajectory", "calculations")


def prepare_feature_calc_directory(method_name: str) -> None:
    create_directory(method_name, "calculations")
    create_directory("transformed", f"calculations/{method_name}")
    create_directory("clusters", f"calculations/{method_name}")
