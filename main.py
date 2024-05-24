import argparse
import os

from prepare_trajectory import prepare_trajectory
from prepare_filestructure import prepare_main_calc_directory, prepare_feature_calc_directory
from features_config import get_manager
from prepare_features import FeaturesManager
from tsne_manager import TSNEManager


def main():
    parser = argparse.ArgumentParser(
        description="Clustering metadynamic calculation")

    parser.add_argument(
        "topology", help="Path to the topology file [.pdb]", type=str)

    parser.add_argument(
        "trajectory", help="Path to trajectory file [.xtc]", type=str)

    parser.add_argument(
        "weights", help="Path to weighting file [PLUMED-compatible]", type=str)

    parser.add_argument(
        "weights_field", help="The name of the variable responsible for the weights in `weights`", type=str)

    parser.add_argument(
        "num_frames", help="Number of frames for clustering", type=int)

    parser.add_argument(
        "-lj", help="Path to Lennard-Jones energies file [GROMACS-compatible]", default=None, type=str)

    # parser.add_argument(
    #     "-num_proc", help="The maximum number of processes in which calculations can be run. Enter 0 to use all available.", type=int, default=1
    # )
    # if parser.parse_args().num_proc == 0:
    #     parser.parse_args().num_proc = os.cpu_count()

    assert parser.parse_args().topology.endswith(
        ".pdb"), "The topology file must be in .pdb format"
    assert parser.parse_args().trajectory.endswith(
        ".xtc"), "The trajectory file must be in .xtc format"
    with open(parser.parse_args().weights) as f:
        assert f.readline().startswith(
            "#! FIELDS"), "The weighting coefficient file must be in a PLUMED-compatible format"
    with open(parser.parse_args().lj) as f:
        for _ in range(2):
            f.readline()
        assert "GROMACS" in f.readline(
        ), "The Lennard-Jones energy file must be in a GROMACS-compatible format"

    args = parser.parse_args()

    # подготовка директории расчетов
    prepare_main_calc_directory()

    prepare_trajectory(
        topology=args.topology,
        trajectory=args.trajectory,
        weights=args.weights,
        weights_field=args.weights_field,
        num_frames=args.num_frames
    )

    tsne_manager = TSNEManager("tsne_log.csv")
    tsne_manager.init_file()

    features_manager: FeaturesManager = get_manager(
        # num_proc=args.num_proc,
        manager=tsne_manager,
        topology_filename=args.topology,
        trajectory_filename="calculations/trajectory/frames.xtc",
        lj_filename=args.lj,
        frames_filename="calculations/trajectory/frames.ndx"
    )

    features_manager.prepare_features()
    features_manager.proceed_clustering()

    print("Done! Watch log file.")


if __name__ == "__main__":
    main()
