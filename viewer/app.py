import sys
from pathlib import Path

import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "raw" / "velodyne_points" / "data"

from adaptive_sampling import adaptive_sample

st.set_page_config(
    page_title="Adaptive LiDAR Lab",
    layout="wide"
)

st.title("Adaptive LiDAR Lab")
st.caption("Geometry-aware point-cloud reduction and evaluation")

@st.cache_data
def load_frame(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]

@st.cache_data
def reduce_frame(path, voxel_size, target):
    xyz = load_frame(path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    uniform = np.asarray(
        pcd.voxel_down_sample(voxel_size).points
    )
    adaptive = adaptive_sample(
        xyz, target_points=min(target, len(xyz))
    )

    return xyz, uniform, adaptive

def make_plot(points, title, max_points=20000):
    # Limit only the displayed points for browser performance.
    # Evaluation should use the complete point clouds.
    if len(points) > max_points:
        rng = np.random.default_rng(42)
        ids = rng.choice(len(points), max_points, replace=False)
        points = points[ids]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=points[:, 2],
                        colorscale="Turbo", opacity=0.8)
        )
    ])

    fig.update_layout(
        title=title,
        height=550,
        margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data"
        )
    )
    return fig


def reconstruction_metrics(original, reduced):
    distances, _ = cKDTree(reduced).query(original, k=1)

    return {
        "Mean error (m)": float(np.mean(distances)),
        "P95 error (m)": float(np.percentile(distances, 95)),
    }


def neighborhood_recall(original, reduced, k=10, tolerance=0.3):
    rng = np.random.default_rng(42)
    ids = rng.choice(
        len(original), min(1000, len(original)), replace=False
    )
    sample = original[ids]

    _, neighbors = cKDTree(original).query(sample, k=k + 1)
    neighbor_points = original[neighbors[:, 1:]].reshape(-1, 3)

    distances, _ = cKDTree(reduced).query(neighbor_points, k=1)
    return float(np.mean(distances <= tolerance))


def distance_metrics(original, reduced):
    ranges = np.linalg.norm(original, axis=1)
    distances, _ = cKDTree(reduced).query(original, k=1)

    rows = []
    bands = [(0, 10), (10, 20), (20, 40), (40, np.inf)]

    for low, high in bands:
        mask = (ranges >= low) & (ranges < high)

        if mask.any():
            rows.append({
                "Distance band": f"{low}-{high if np.isfinite(high) else '+'} m",
                "Original points": int(mask.sum()),
                "Mean error (m)": float(distances[mask].mean()),
                "P95 error (m)": float(np.percentile(distances[mask], 95)),
            })

    return pd.DataFrame(rows)


files = sorted(DATA_DIR.glob("*.bin"))

if not files:
    st.error(f"No KITTI frames found in {DATA_DIR}")
    st.stop()

frame = st.selectbox(
    "Select LiDAR frame",
    files,
    format_func=lambda p: p.name
)

voxel_size = st.slider(
    "Uniform voxel size (m)", 0.2, 1.2, 0.8, 0.1
)

if st.button("Compare methods", type="primary"):
    with st.spinner("Processing point cloud..."):
        xyz = load_frame(str(frame))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        uniform = np.asarray(
            pcd.voxel_down_sample(voxel_size).points
        )

        adaptive = adaptive_sample(
            xyz, target_points=len(uniform)
        )

    st.session_state["comparison"] = (
        xyz, uniform, adaptive, frame.name
    )

if "comparison" in st.session_state:
    xyz, uniform, adaptive, frame_name = (
        st.session_state["comparison"]
    )

    st.subheader(f"Frame: {frame_name}")

    cols = st.columns(3)
    clouds = [xyz, uniform, adaptive]
    names = ["Original", "Uniform baseline", "Adaptive"]

    for col, cloud, name in zip(cols, clouds, names):
        with col:
            st.metric("Points", f"{len(cloud):,}")
            st.plotly_chart(
                make_plot(cloud, name),
                use_container_width=True
            )

    st.subheader("Point reduction")

    c1, c2 = st.columns(2)
    c1.metric(
        "Uniform reduction",
        f"{100 * (1 - len(uniform) / len(xyz)):.2f}%"
    )
    c2.metric(
        "Adaptive reduction",
        f"{100 * (1 - len(adaptive) / len(xyz)):.2f}%"
    )

    
    st.download_button(
        "Download adaptive cloud (.npy)",
        data=adaptive.astype(np.float32).tobytes(),
        file_name="adaptive_points.bin",
        mime="application/octet-stream"
    )

    # STEP 2: Quantitative evaluation
    st.divider()
    st.header("Quantitative Evaluation")

    uniform_metrics = reconstruction_metrics(xyz, uniform)
    adaptive_metrics = reconstruction_metrics(xyz, adaptive)

    uniform_recall = neighborhood_recall(xyz, uniform)
    adaptive_recall = neighborhood_recall(xyz, adaptive)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Adaptive mean error",
        f"{adaptive_metrics['Mean error (m)']:.3f} m",
        delta=f"{adaptive_metrics['Mean error (m)'] - uniform_metrics['Mean error (m)']:.3f} m vs uniform",
        delta_color="inverse",
    )

    c2.metric(
        "Adaptive P95 error",
        f"{adaptive_metrics['P95 error (m)']:.3f} m",
        delta=f"{adaptive_metrics['P95 error (m)'] - uniform_metrics['P95 error (m)']:.3f} m vs uniform",
        delta_color="inverse",
    )

    c3.metric(
        "Adaptive neighbourhood recall",
        f"{adaptive_recall:.1%}",
        delta=f"{(adaptive_recall - uniform_recall) * 100:.2f} percentage points",
    )

    st.subheader("Error by distance")

    uniform_distance = distance_metrics(xyz, uniform)
    adaptive_distance = distance_metrics(xyz, adaptive)

    distance_table = uniform_distance.merge(
        adaptive_distance,
        on=["Distance band", "Original points"],
        suffixes=(" (Uniform)", " (Adaptive)"),
    )

    st.dataframe(distance_table, use_container_width=True)
    
    st.subheader("Reconstruction Error by Distance")

    chart_data = distance_table[
        [
            "Distance band",
            "Mean error (m) (Uniform)",
            "Mean error (m) (Adaptive)",
        ]
    ].melt(
        id_vars="Distance band",
        var_name="Method",
        value_name="Mean error (m)",
    )

    chart_data["Method"] = chart_data["Method"].replace({
        "Mean error (m) (Uniform)": "Uniform baseline",
        "Mean error (m) (Adaptive)": "Adaptive",
    })

    import plotly.express as px

    fig = px.bar(
        chart_data,
        x="Distance band",
        y="Mean error (m)",
        color="Method",
        barmode="group",
        title="Mean Reconstruction Error by Distance",
        labels={
            "Distance band": "Distance from LiDAR sensor",
            "Mean error (m)": "Mean nearest-neighbour error (m)",
        },
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        legend_title_text="Sampling method",
    )

    st.plotly_chart(fig, use_container_width=True)


    st.download_button(
        "Download distance evaluation CSV",
        data=distance_table.to_csv(index=False),
        file_name="distance_evaluation.csv",
        mime="text/csv",
    )

