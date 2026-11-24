
import numpy as np
from adaptive_sampling import adaptive_sample_v4


def run_test(name, points, target_points, base_voxel_size=0.4):
    print(f"\n--- {name} ---")

    sampled = adaptive_sample_v4(
        points,
        target_points=target_points,
        base_voxel_size=base_voxel_size
    )

    print("Input points:", len(points))
    print("Output points:", len(sampled))
    print("Output shape:", sampled.shape)

    assert sampled.ndim == 2
    assert sampled.shape[1] == 3
    assert len(sampled) == min(target_points, len(points))
    assert np.isfinite(sampled).all()

    # Every returned point must exist in the input.
    # This checks exact coordinate matches.
    input_points = {tuple(p) for p in points}
    assert all(tuple(p) in input_points for p in sampled)

    print("PASS")


if __name__ == "__main__":
    np.random.seed(42)

    # Test 1: Synthetic point cloud
    points = np.random.uniform(
        low=[0, -20, -3],
        high=[50, 20, 3],
        size=(125100, 3)
    ).astype(np.float32)

    run_test("Normal sampling", points, 9416)

    # Test 2: Target larger than input
    small_points = points[:100]
    run_test("Target exceeds input", small_points, 200)

    # Test 3: Target equals input
    run_test("Target equals input", small_points, 100)

    # Test 4: Zero target
    run_test("Zero target", small_points, 0)

    # Test 5: Empty input
    empty_points = np.empty((0, 3), dtype=np.float32)
    run_test("Empty input", empty_points, 10)

    print("\nAll tests passed!")
