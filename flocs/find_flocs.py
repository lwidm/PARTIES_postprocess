# -- flocs/find_flocs.py

import numpy as np
from scipy.spatial import cKDTree  # type: ignore
from typing import Dict, Union, List, Set


def find_flocs(
    particle_data: Dict[str, np.ndarray],
    cutoff_distance: float,
    domain: Dict[str, Union[int, float]],
) -> Dict[str, np.ndarray]:
    """Find all flocs for a particle output file.

    See `adjacency_list` for the definition of a floc.

    Args:
        parameter_data: Dict containing particle cooridantes and radii.
        cutoff_distance: Distance between particle surfaces within which particles are
                     considered "adjacent".

    Returns:
        Particle data with an additional column, `floc_id`, indicating which floc the
        particle is a part of.
    """
    adjacency = _make_adj_list(particle_data, cutoff_distance, domain)
    flocs = _find_connected_groups(adjacency)
    particle_data = _assign_floc_ids(particle_data, flocs)
    return particle_data


def _assign_floc_ids(
    particle_data: Dict[str, np.ndarray], flocs: List[List[int]]
) -> dict[str, np.ndarray]:
    """Covert floc data from indexed by floc-id to indexed by particle-id.

    Args:
        particle_data: Dict contaning particle data.
        flocs: List of lists containing particle-ids which form a connected group (a floc).

    Returns:
        particle data dict with an additional key, `floc_id`.
    """
    floc_ids: np.ndarray = np.zeros_like(particle_data["r"], dtype=int)
    floc_id: int
    floc: List[int]
    for floc_id, floc in enumerate(flocs):
        floc_ids[floc] = floc_id
    particle_data["floc_id"] = floc_ids
    return particle_data


def _make_adj_list(
    particle_data: dict[str, np.ndarray],
    cutoff_distance: float,
    domain: Dict[str, Union[int, float]],
) -> dict[int, np.ndarray]:
    """Compute the adjacency "list" (dict), accounting for periodic boundaries.

    Compute the adjacency list given the following condition is true:

        ||X_1 - X_2|| <= r_1 + r_2 + cutoff_dist

    Args:
        particle_data: Dict containing particle coordinates and radii.
        cutoff_distance: Distance between particle surfaces within which particles are
                     considered "adjacent".
        domain: Information on domain size and periodicity in each direction.

    Returns:
        Adjacency "list" (dict) where the key is the particle-id and entries are lists
        of particle-ids of the neighboring particles.
    """
    X_p: np.ndarray = np.column_stack(
        (particle_data["x"], particle_data["y"], particle_data["z"])
    )
    n_p: int = len(particle_data["r"])
    adj: Dict[int, np.ndarray] = {i: np.array([]) for i in range(n_p)}

    # Handle periodic boundary conditions
    offsets: np.ndarray = np.array([[0, 0, 0]])
    if domain["x_periodic"]:
        offsets = np.vstack([offsets, [domain["Lx"], 0, 0], [-domain["Lx"], 0, 0]])
    if domain["y_periodic"]:
        offsets = np.vstack([offsets, [0, domain["Ly"], 0], [0, -domain["Ly"], 0]])
    if domain["z_periodic"]:
        offsets = np.vstack([offsets, [0, 0, domain["Lz"]], [0, 0, -domain["Lz"]]])
    X_p_extended: np.ndarray = np.vstack([X_p + offset for offset in offsets])

    tree = cKDTree(X_p_extended)
    i: int
    particle: np.ndarray
    for i, particle in enumerate(X_p):
        neighbors: List[int] = tree.query_ball_point(
            particle, 2 * particle_data["r"][i] + cutoff_distance
        )
        neighbors.remove(i)
        # Re-collapse the adjacency data to only include indices from the original dataset
        neighbors = [n % n_p for n in neighbors]
        adj[i] = np.array(neighbors)

    return adj


def _find_connected_groups(adj: Dict[int, np.ndarray]) -> List[List[int]]:
    """Find the flocs within an adjacency "list" (dict).

    Recursively find all the particles that are connected, including through neighbors.

    Args:
        adj: Adjacency "list" (dict). Output of `compute_adjacency`.

    Returns:
        List containing lists of the particle-ids which form a connected group (a floc).
    """
    visited: Set[int] = set()
    connected_groups: List[List[int]] = []

    # depth-first search
    def dfs(node: int, group: List[int]):
        visited.add(node)
        group.append(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor, group)

    node: int
    for node in adj:
        if node not in visited:
            connected_group: List[int] = []
            dfs(node, connected_group)
            connected_groups.append(connected_group)

    return connected_groups
