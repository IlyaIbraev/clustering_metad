import os


def prepare_main_calc_directory() -> None:
    if "calculations" not in os.listdir():
        os.mkdir(
            "calculations"
        )
    if "trajectory" not in os.listdir("calculations"):
        os.mkdir(
            "calculations/trajectory"
        )


def prepare_feature_calc_directory(method_name: str) -> None:
    if method_name not in os.listdir("calculations"):
        os.mkdir(
            f"calculations/{method_name}"
        )
    if "transformed" not in os.listdir(f"calculations/{method_name}"):
        os.mkdir(
            f"calculations/{method_name}/transformed"
        )
    if "clusters" not in os.listdir(f"calculations/{method_name}"):
        os.mkdir(
            f"calculations/{method_name}/clusters"
        )
