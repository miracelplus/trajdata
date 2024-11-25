import os
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Any
from trajdata import UnifiedDataset, AgentType, SceneBatch
from trajdata.maps.vec_map_elements import (
    MapElement,
    MapElementType,
    PedCrosswalk,
    RoadLane,
)

# Constants for data format
_agent_types = ["vehicle", "pedestrian", "cyclist", "background"]
_polygon_types = ["VEHICLE", "BIKE", "BUS", "PEDESTRIAN"]
_polygon_light_type = [
    "LANE_STATE_STOP",
    "LANE_STATE_GO",
    "LANE_STATE_CAUTION",
    "LANE_STATE_UNKNOWN",
]
_point_types = [
    "DASH_SOLID_YELLOW",
    "DASH_SOLID_WHITE",
    "DASHED_WHITE",
    "DASHED_YELLOW",
    "DOUBLE_SOLID_YELLOW",
    "DOUBLE_SOLID_WHITE",
    "DOUBLE_DASH_YELLOW",
    "DOUBLE_DASH_WHITE",
    "SOLID_YELLOW",
    "SOLID_WHITE",
    "SOLID_DASH_WHITE",
    "SOLID_DASH_YELLOW",
    "EDGE",
    "NONE",
    "UNKNOWN",
    "CROSSWALK",
    "CENTERLINE",
]


# def process_agent_batch(batch: SceneBatch) -> Dict[str, Any]:


def process_agent_batch(
    batch: SceneBatch, hist_len: int = 12, fut_len: int = 80
) -> Dict[str, Any]:
    """Process agent data from TrajData SceneBatch to SMART format.

    Args:
        batch: SceneBatch containing scene and agent information
        hist_len: Fixed number of historical timesteps
        fut_len: Fixed number of future timesteps

    Returns:
        Dictionary containing processed agent data in SMART format
    """
    print("agent_hist shape:", batch.agent_hist.shape)
    print("agent_fut shape:", batch.agent_fut.shape)
    print("agent_hist_len:", batch.agent_hist_len)
    print("agent_fut_len:", batch.agent_fut_len)
    num_agents = batch.num_agents  # tensor([29])
    total_steps = hist_len + fut_len  # 固定总时间步长

    # Initialize tensors
    position = torch.zeros(num_agents, total_steps, 3)
    heading = torch.zeros(num_agents, total_steps)
    velocity = torch.zeros(num_agents, total_steps, 3)
    shape = torch.zeros(num_agents, total_steps, 3)

    # Initialize masks
    valid_mask = torch.zeros(num_agents, total_steps, dtype=torch.bool)
    predict_mask = torch.zeros(num_agents, total_steps, dtype=torch.bool)

    # Fill data for each agent
    for i in range(num_agents):
        # Get actual lengths for this agent
        agent_hist_len = min(batch.agent_hist_len[0, i].item(), hist_len)
        agent_fut_len = min(batch.agent_fut_len[0, i].item(), fut_len)

        # Fill history data
        if agent_hist_len > 0:
            position[i, hist_len - agent_hist_len : hist_len] = batch.agent_hist[
                0, i, -agent_hist_len:, :3
            ]
            heading[i, hist_len - agent_hist_len : hist_len] = batch.agent_hist[
                0, i, -agent_hist_len:, 5
            ]
            velocity[i, hist_len - agent_hist_len : hist_len] = batch.agent_hist[
                0, i, -agent_hist_len:, 2:5
            ]
            shape[i, hist_len - agent_hist_len : hist_len] = batch.agent_hist_extent[
                0, i, -agent_hist_len:
            ]
            valid_mask[i, hist_len - agent_hist_len : hist_len] = True

        # Fill future data
        if agent_fut_len > 0:
            position[i, hist_len : hist_len + agent_fut_len] = batch.agent_fut[
                0, i, :agent_fut_len, :3
            ]
            heading[i, hist_len : hist_len + agent_fut_len] = batch.agent_fut[
                0, i, :agent_fut_len, 5
            ]
            velocity[i, hist_len : hist_len + agent_fut_len] = batch.agent_fut[
                0, i, :agent_fut_len, 2:5
            ]
            shape[i, hist_len : hist_len + agent_fut_len] = batch.agent_fut_extent[
                0, i, :agent_fut_len
            ]
            valid_mask[i, hist_len : hist_len + agent_fut_len] = True

    # Set prediction mask (all future steps, regardless of validity)
    predict_mask[:, hist_len:] = True

    return {
        "num_nodes": num_agents[0],  # Convert from tensor to int
        "av_index": 0,  # Assuming ego vehicle is at index 0
        "valid_mask": valid_mask,
        "predict_mask": predict_mask,
        "id": batch.agent_names[0],
        "type": batch.agent_type[0],
        "category": batch.agent_type.clone(),
        "position": position,
        "heading": heading,
        "velocity": velocity,
        "shape": shape,
    }


def process_map_features(batch, dim: int = 3) -> Dict[str, Any]:
    """Process map features from TrajData batch to SMART format.

    Args:
        batch: TrajData batch containing map information
        dim: Spatial dimension of the data (default: 3)

    Returns:
        Dictionary containing processed map data in SMART format
    """
    # Try to get vector map directly from batch
    vector_map = batch.vector_maps[0]
    if vector_map is None:
        return {}

    # Initialize lists to store all map elements
    all_polygons = []
    polygon_types = []
    polygon_light_types = []

    # Process lanes
    if hasattr(vector_map, "lanes"):
        for lane_id, lane in vector_map.elements[MapElementType.ROAD_LANE].items():
            points = lane.center.points
            all_polygons.append(points)
            polygon_types.append(0)  # Assuming 0 for lanes

            # Get traffic light status if available
            light_state = 3  # Default UNKNOWN state
            if vector_map.traffic_light_status:
                # Method 1: Get all timesteps for this specific lane_id
                timesteps = set(
                    ts
                    for lid, ts in vector_map.traffic_light_status.keys()
                    if lid == lane_id
                )
                if timesteps:
                    # If timesteps found, use the first state
                    ts = min(timesteps)  # or use next(iter(timesteps))
                    light_state = vector_map.traffic_light_status[(lane_id, ts)].value

                # Method 2: Get all possible timestep range (alternative approach)
                # max_ts = max(ts for _, ts in vector_map.traffic_light_status.keys())
                # for ts_idx in range(max_ts + 1):
                #     if (lane_id, ts_idx) in vector_map.traffic_light_status:
                #         light_state = vector_map.traffic_light_status[(lane_id, ts_idx)].value
                #         break

            polygon_light_types.append(light_state)

    # Process crosswalks
    if (
        hasattr(vector_map, "elements")
        and MapElementType.PED_CROSSWALK in vector_map.elements
    ):
        for crosswalk in vector_map.elements[MapElementType.PED_CROSSWALK].values():
            points = crosswalk.polygon.points  # Should be Nx3 array
            all_polygons.append(points)
            polygon_types.append(1)  # Assuming 1 for crosswalks
            polygon_light_types.append(3)  # Default no light

    # If no map elements found, return empty dict
    if not all_polygons:
        return {}

    # Convert lists to tensors
    num_elements = len(all_polygons)
    polygon_type = torch.tensor(polygon_types, dtype=torch.uint8)
    polygon_light_type = torch.tensor(polygon_light_types, dtype=torch.uint8)

    # Create map point features from all polygon points
    all_points = torch.cat([torch.tensor(poly) for poly in all_polygons], dim=0)

    return {
        "map_polygon": {
            "num_nodes": num_elements,
            "type": polygon_type,
            "light_type": polygon_light_type,
        },
        "map_point": {
            "num_nodes": len(all_points),
            "position": all_points,
            "orientation": compute_point_orientations(
                all_points
            ),  # Need to implement this
            "magnitude": torch.ones(len(all_points)) * 0.5,  # Default magnitude
            "height": torch.zeros(len(all_points)),  # Default height
            "type": torch.full(
                (len(all_points),), 2, dtype=torch.uint8
            ),  # Default type
        },
    }


def compute_point_orientations(points: torch.Tensor) -> torch.Tensor:
    """Compute orientation angles for each point based on its neighbors.

    Args:
        points: Nx3 tensor of point coordinates

    Returns:
        N tensor of orientation angles in radians
    """
    # For each point except endpoints, compute angle between previous and next point
    orientations = torch.zeros(len(points))

    # For points with neighbors, compute orientation from vector between adjacent points
    for i in range(1, len(points) - 1):
        prev_pt = points[i - 1, :2]
        next_pt = points[i + 1, :2]
        vector = next_pt - prev_pt
        angle = torch.atan2(vector[1], vector[0])
        orientations[i] = angle

    # For endpoints, use vector to/from neighbor
    if len(points) > 1:
        orientations[0] = torch.atan2(
            points[1, 1] - points[0, 1], points[1, 0] - points[0, 0]
        )
        orientations[-1] = torch.atan2(
            points[-1, 1] - points[-2, 1], points[-1, 0] - points[-2, 0]
        )

    return orientations


def convert_batch_to_smart(batch, output_dir: str):
    """Convert a single TrajData batch to SMART format and save it.

    Args:
        batch: TrajData batch to convert
        output_dir: Directory to save the converted data
    """
    # Process agent data
    agent_data = process_agent_batch(batch)

    # Process map data
    map_data = process_map_features(batch)

    # Combine all data
    data = {
        "scenario_id": batch.scene_ids[0],
        "city": "unknown",  # TrajData doesn't have city information
        "agent": agent_data,
    }
    data.update(map_data)

    # Save to pickle file
    output_path = os.path.join(output_dir, f"{data['scenario_id']}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def main():
    """Main function to run the conversion process."""
    # Create UnifiedDataset
    dataset = UnifiedDataset(
        desired_data=["waymo_val"],  # or other datasets
        data_dirs={  # Remember to change this to match your filesystem!
            "nuplan_mini": "/home/haoweis/trajdata_smart/trajdata/data/nuplan/dataset/nuplan-v1.1/",
            "waymo_val": "/home/haoweis/trajdata_smart/trajdata/data/waymo/",
        },
        centric="scene",
        desired_dt=0.1,
        # history_sec=(1.0, 1.0),  # 10 frames of history
        # future_sec=(8.0, 8.0),  # 80 frames of future
        only_predict=[AgentType.VEHICLE],
        state_format="x,y,z,xd,yd,h",
        obs_format="x,y,z,xd,yd,s,c",
        incl_robot_future=False,
        incl_raster_map=False,
        incl_vector_map=True,
        vector_map_params={
            "incl_road_lanes": True,
            "incl_road_areas": True,
            "incl_ped_crosswalks": True,
            "collate": True,
        },
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one scene at a time
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )

    # Create output directory
    output_dir = "data/smart_format"
    os.makedirs(output_dir, exist_ok=True)

    # Process data
    for batch in tqdm(dataloader, desc="Converting to SMART format"):
        convert_batch_to_smart(batch, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert TrajData to SMART format")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/smart_format",
        help="Output directory for converted data",
    )
    parser.add_argument(
        "--dataset", type=str, default="waymo_train", help="Dataset to convert"
    )
    args = parser.parse_args()

    main()
