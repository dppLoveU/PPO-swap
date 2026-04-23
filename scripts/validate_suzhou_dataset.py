import argparse
import os
import pickle
import sys

import networkx as nx
import numpy as np


def parse_city_num(data_path):
    name = data_path.rstrip("/\\").split("/")[-1].split("\\")[-1]
    try:
        return int(name.split("_")[-1])
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse sample count from data_path name: {name!r}. "
            "The directory name must end with _<sample_count>, e.g. suzhou_grid_3."
        ) from exc


def parse_max_p(fac_range, max_p):
    if fac_range is not None:
        try:
            values = list(eval(fac_range, {"__builtins__": {}}, {"range": range}))
        except Exception as exc:
            raise ValueError(f"Invalid fac_range: {fac_range!r}") from exc
        if not values:
            raise ValueError(f"fac_range produced no values: {fac_range!r}")
        return max(values)
    return max_p


def is_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(value)


def validate_pos(node_id, attrs, errors):
    has_pos = "pos" in attrs
    has_xy = "x" in attrs and "y" in attrs

    if not has_pos and not has_xy:
        errors.append(f"node {node_id} missing pos or x/y")
        return

    if has_pos:
        pos = attrs["pos"]
        try:
            pos_arr = np.asarray(pos, dtype=float)
        except (TypeError, ValueError):
            errors.append(f"node {node_id} pos is not numeric: {pos!r}")
            return
        if pos_arr.shape != (2,):
            errors.append(f"node {node_id} pos must have shape (2,), got {pos_arr.shape}")
        elif not np.isfinite(pos_arr).all():
            errors.append(f"node {node_id} pos contains NaN or inf")
        return

    if not is_number(attrs["x"]):
        errors.append(f"node {node_id} x is not finite numeric: {attrs['x']!r}")
    if not is_number(attrs["y"]):
        errors.append(f"node {node_id} y is not finite numeric: {attrs['y']!r}")


def validate_sample(sample_dir, sample_id, max_p=None, check_symmetric=False):
    errors = []
    graph_path = os.path.join(sample_dir, "graph.pkl")
    dist_path = os.path.join(sample_dir, "distance_m.pkl")

    if not os.path.isfile(graph_path):
        errors.append(f"missing graph.pkl: {graph_path}")
    if not os.path.isfile(dist_path):
        errors.append(f"missing distance_m.pkl: {dist_path}")
    if errors:
        return None, errors

    try:
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
    except Exception as exc:
        errors.append(f"cannot load graph.pkl: {exc}")
        return None, errors

    try:
        with open(dist_path, "rb") as f:
            distance_m = np.asarray(pickle.load(f), dtype=float)
    except Exception as exc:
        errors.append(f"cannot load distance_m.pkl: {exc}")
        return None, errors

    if not isinstance(graph, nx.Graph):
        errors.append(f"graph.pkl must be a NetworkX graph, got {type(graph)!r}")

    node_ids = list(graph.nodes())
    n = len(node_ids)
    expected_nodes = list(range(n))
    sorted_nodes = sorted(node_ids)
    if sorted_nodes != expected_nodes:
        missing = sorted(set(expected_nodes) - set(sorted_nodes))
        extra = sorted(set(sorted_nodes) - set(expected_nodes))
        errors.append(
            "node ids must be continuous integers 0..n-1; "
            f"missing={missing}, extra={extra}, actual_head={sorted_nodes[:10]}"
        )

    pop_values = []
    coord_values = []
    for node_id, attrs in graph.nodes(data=True):
        if "pop" not in attrs:
            errors.append(f"node {node_id} missing pop")
        elif not is_number(attrs["pop"]):
            errors.append(f"node {node_id} pop is not finite numeric: {attrs['pop']!r}")
        else:
            pop = float(attrs["pop"])
            pop_values.append(pop)
            if pop < 0:
                errors.append(f"node {node_id} pop is negative: {pop}")

        validate_pos(node_id, attrs, errors)
        if "pos" in attrs:
            try:
                coord_values.append(np.asarray(attrs["pos"], dtype=float))
            except (TypeError, ValueError):
                pass
        elif "x" in attrs and "y" in attrs and is_number(attrs["x"]) and is_number(attrs["y"]):
            coord_values.append(np.asarray([attrs["x"], attrs["y"]], dtype=float))

    if pop_values and sum(pop_values) <= 0:
        errors.append("sum of node pop must be > 0")

    if len(coord_values) == n and n > 0:
        coords = np.vstack(coord_values)
        span = coords.max(axis=0) - coords.min(axis=0)
        if np.max(span) <= 0:
            errors.append("coordinates must have non-zero span in x or y")

    for u, v, attrs in graph.edges(data=True):
        if "length" not in attrs:
            errors.append(f"edge {(u, v)} missing length")
            continue
        if not is_number(attrs["length"]):
            errors.append(f"edge {(u, v)} length is not finite numeric: {attrs['length']!r}")
            continue
        length = float(attrs["length"])
        if length <= 0:
            errors.append(f"edge {(u, v)} length must be > 0, got {length}")

    if distance_m.ndim != 2:
        errors.append(f"distance_m must be 2D, got shape {distance_m.shape}")
    elif distance_m.shape != (n, n):
        errors.append(f"distance_m shape must be ({n}, {n}), got {distance_m.shape}")
    else:
        if np.isnan(distance_m).any():
            errors.append("distance_m contains NaN")
        if np.isinf(distance_m).any():
            errors.append("distance_m contains inf")
        if distance_m.size > 0 and np.isfinite(distance_m).all() and distance_m.max() <= 0:
            errors.append("distance_m.max() must be > 0")
        diag = np.diag(distance_m)
        if not np.allclose(diag, 0):
            errors.append("distance_m diagonal should be 0 or close to 0")
        if check_symmetric and not np.allclose(distance_m, distance_m.T):
            errors.append("distance_m is not symmetric")

    if max_p is not None and max_p > n:
        errors.append(f"max_p={max_p} exceeds node_count={n}")

    summary = {
        "sample_id": sample_id,
        "nodes": n,
        "edges": graph.number_of_edges() if isinstance(graph, nx.Graph) else None,
        "distance_shape": distance_m.shape,
    }
    return summary, errors


def validate_dataset(data_path, max_p=None, check_symmetric=False):
    city_num = parse_city_num(data_path)
    all_errors = []
    summaries = []

    for sample_id in range(city_num):
        sample_dir = os.path.join(data_path, str(sample_id))
        if not os.path.isdir(sample_dir):
            all_errors.append(f"sample {sample_id}: missing directory {sample_dir}")
            continue
        summary, errors = validate_sample(
            sample_dir,
            sample_id,
            max_p=max_p,
            check_symmetric=check_symmetric,
        )
        if summary is not None:
            summaries.append(summary)
        for error in errors:
            all_errors.append(f"sample {sample_id}: {error}")

    return city_num, summaries, all_errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate Suzhou PPO-swap dataset graph.pkl and distance_m.pkl files."
    )
    parser.add_argument("--data-path", required=True, help="Dataset path, e.g. data/suzhou_grid_3/")
    parser.add_argument(
        "--fac-range",
        default=None,
        help='Facility range string, e.g. "range(5, 16, 5)". Used to check max p <= node count.',
    )
    parser.add_argument(
        "--max-p",
        type=int,
        default=None,
        help="Maximum facility count. Ignored if --fac-range is provided.",
    )
    parser.add_argument(
        "--check-symmetric",
        action="store_true",
        help="Also check whether distance_m is symmetric.",
    )
    args = parser.parse_args()

    try:
        max_p = parse_max_p(args.fac_range, args.max_p)
        city_num, summaries, errors = validate_dataset(
            args.data_path,
            max_p=max_p,
            check_symmetric=args.check_symmetric,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2

    print(f"[INFO] data_path: {args.data_path}")
    print(f"[INFO] city_num: {city_num}")
    if max_p is not None:
        print(f"[INFO] max_p: {max_p}")

    for summary in summaries:
        print(
            "[OK] sample {sample_id}: nodes={nodes}, edges={edges}, "
            "distance_m={distance_shape}".format(**summary)
        )

    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        print("Validation failed.")
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
