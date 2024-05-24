
from prepare_features import Feature, FeaturesManager
from tsne_manager import TSNEManager

from features.residue_distance import prepare_matrix as residue_distance_prepare_matrix
from features.lennard_jones import prepare_matrix as lennard_jones_prepare_matrix
from features.heavy_atoms import prepare_matrix as heavy_atoms_prepare_matrix


def get_manager(
    #     num_proc: int,
    manager: TSNEManager,
    topology_filename: str,
    trajectory_filename: str,
    lj_filename: str,
    frames_filename: str
) -> FeaturesManager:

    features_manager = FeaturesManager(
        # num_proc=num_proc,
        manager=manager
    )

    features_manager.add_feature(
        Feature(
            name="residue_distance",
            prepare_function=residue_distance_prepare_matrix,
            params={
                "topology_filename": topology_filename,
                "trajectory_filename": trajectory_filename,
            }
        )
    )

    if lj_filename != None:
        features_manager.add_feature(
            Feature(
                name="lennard_jones",
                prepare_function=lennard_jones_prepare_matrix,
                params={
                    "lj_filename": lj_filename,
                    "frames_filename": frames_filename
                }
            )
        )

    features_manager.add_feature(
        Feature(
            name="heavy_atoms",
            prepare_function=heavy_atoms_prepare_matrix,
            params={
                "topology_filename": topology_filename,
                "trajectory_filename": trajectory_filename,
            }
        )
    )

    return features_manager
