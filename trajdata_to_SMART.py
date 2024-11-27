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

_polygon_to_polygon_types = ["NONE", "PRED", "SUCC", "LEFT", "RIGHT"]
# def process_agent_batch(batch: SceneBatch) -> Dict[str, Any]:


def process_agent_batch(
    batch: SceneBatch, hist_len: int = 11, fut_len: int = 80
) -> Dict[str, Any]:
    """Process agent data from TrajData SceneBatch to SMART format.

    Args:
        batch: SceneBatch containing scene and agent information
        hist_len: Fixed number of historical timesteps
        fut_len: Fixed number of future timesteps

    Returns:
        Dictionary containing processed agent data in SMART format
    """
    # print("agent_hist shape:", batch.agent_hist.shape)
    # print("agent_fut shape:", batch.agent_fut.shape)
    # print("agent_hist_len:", batch.agent_hist_len)
    # print("agent_fut_len:", batch.agent_fut_len)
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
            position[i, hist_len - agent_hist_len : hist_len] = (
                batch.agent_hist.position3d[0, i, -agent_hist_len:]
            )
            heading[i, hist_len - agent_hist_len : hist_len] = batch.agent_hist.heading[
                0, i, -agent_hist_len:
            ].squeeze(-1)
            velocity[i, hist_len - agent_hist_len : hist_len, :2] = (
                batch.agent_hist.velocity[0, i, -agent_hist_len:, :2]
            )  # leave the last dim vz to be 0
            shape[i, hist_len - agent_hist_len : hist_len] = batch.agent_hist_extent[
                0, i, -agent_hist_len:
            ]
            valid_mask[i, hist_len - agent_hist_len : hist_len] = True

        # Fill future data
        if agent_fut_len > 0:
            position[i, hist_len : hist_len + agent_fut_len] = (
                batch.agent_fut.position3d[0, i, :agent_fut_len]
            )
            heading[i, hist_len : hist_len + agent_fut_len] = batch.agent_fut.heading[
                0, i, :agent_fut_len
            ].squeeze(-1)
            velocity[i, hist_len : hist_len + agent_fut_len, :2] = (
                batch.agent_fut.velocity[0, i, :agent_fut_len, :2]
            )  # leave the last dim vz to be 0
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


def process_map_features(
    batch, dim: int = 3, target_timestep: int = 11
) -> Dict[str, Any]:
    """Process map features from TrajData batch to SMART format.

    Args:
        batch: TrajData batch containing map information
        dim: Spatial dimension of the data (default: 3)
        target_timestep: Target timestep for traffic light status (default: 11)

    Returns:
        Dictionary containing processed map data in SMART format
    """
    vector_map = batch.vector_maps[0]
    if vector_map is None:
        return {}

    # Initialize lists to store all map elements
    all_polygons = []
    polygon_types = []
    polygon_light_types = []
    point_positions = []
    point_orientations = []
    point_magnitudes = []
    point_heights = []
    point_types = []
    polygon_ids = []
    num_points = []

    # Process lanes
    if hasattr(vector_map, "lanes"):
        for lane_id, lane in vector_map.elements[MapElementType.ROAD_LANE].items():
            # Convert points to tensor
            points = torch.from_numpy(lane.center.points[:, :dim]).float()
            all_polygons.append(points)
            polygon_types.append(0)  # Vehicle lane type
            polygon_ids.append(lane_id)
            num_points.append(len(points))

            # Get traffic light status at target timestep
            light_state = 3  # Default UNKNOWN state
            if vector_map.traffic_light_status:
                if (lane_id, target_timestep) in vector_map.traffic_light_status:
                    light_state = vector_map.traffic_light_status[
                        (lane_id, target_timestep)
                    ].value
            polygon_light_types.append(light_state)

            # Process points
            point_positions.append(points[:-1])  # Exclude last point for vectors
            vectors = points[1:] - points[:-1]  # Compute vectors between points

            # Compute orientations from vectors
            orientations = torch.atan2(vectors[:, 1], vectors[:, 0])
            point_orientations.append(orientations)

            # Compute magnitudes (length of x,y vectors)
            magnitudes = torch.norm(vectors[:, :2], p=2, dim=1)
            point_magnitudes.append(magnitudes)

            # Get heights (z component of vectors)
            if dim > 2:
                heights = vectors[:, 2]
                point_heights.append(heights)

            # Set point type as CENTERLINE
            point_types.append(
                torch.full((len(vectors),), 2, dtype=torch.uint8)
            )  # 2 for CENTERLINE

    # Process crosswalks (similar to lanes)
    if (
        hasattr(vector_map, "elements")
        and MapElementType.PED_CROSSWALK in vector_map.elements
    ):
        for crosswalk_id, crosswalk in vector_map.elements[
            MapElementType.PED_CROSSWALK
        ].items():
            # Convert points to tensor
            points = torch.from_numpy(crosswalk.polygon.points[:, :dim]).float()
            all_polygons.append(points)
            polygon_types.append(3)  # Pedestrian type
            polygon_light_types.append(3)  # No light state for crosswalks
            polygon_ids.append(crosswalk_id)
            num_points.append(len(points))

            # Process points
            point_positions.append(points[:-1])
            vectors = points[1:] - points[:-1]

            orientations = torch.atan2(vectors[:, 1], vectors[:, 0])
            point_orientations.append(orientations)

            magnitudes = torch.norm(vectors[:, :2], p=2, dim=1)
            point_magnitudes.append(magnitudes)

            if dim > 2:
                heights = vectors[:, 2]
                point_heights.append(heights)

            point_types.append(
                torch.full((len(vectors),), 16, dtype=torch.uint8)
            )  # 16 for CROSSWALK

    # Convert lists to tensors
    num_polygons = len(all_polygons)
    num_points = torch.tensor(num_points, dtype=torch.long)

    # Create point_to_polygon_edge_index
    point_to_polygon_edge_index = torch.stack(
        [
            torch.arange(num_points.sum(), dtype=torch.long),
            torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points),
        ],
        dim=0,
    )

    # Create polygon_to_polygon_edge_index and polygon_to_polygon_type
    polygon_to_polygon_edge_index = []
    polygon_to_polygon_type = []

    # Process lane connections
    for lane_id, lane in vector_map.elements[MapElementType.ROAD_LANE].items():
        lane_idx = polygon_ids.index(lane_id)

        # Process predecessors
        for pred in lane.prev_lanes:
            pred_idx = safe_list_index(polygon_ids, pred)
            if pred_idx is not None:
                polygon_to_polygon_edge_index.append(
                    torch.tensor([[pred_idx], [lane_idx]], dtype=torch.long)
                )
                polygon_to_polygon_type.append(
                    torch.tensor(
                        [_polygon_to_polygon_types.index("PRED")], dtype=torch.uint8
                    )
                )

        # Process successors
        for succ in lane.next_lanes:
            succ_idx = safe_list_index(polygon_ids, succ)
            if succ_idx is not None:
                polygon_to_polygon_edge_index.append(
                    torch.tensor([[lane_idx], [succ_idx]], dtype=torch.long)
                )
                polygon_to_polygon_type.append(
                    torch.tensor(
                        [_polygon_to_polygon_types.index("SUCC")], dtype=torch.uint8
                    )
                )

        # Process left neighbors
        for left in lane.adj_lanes_left:
            left_idx = safe_list_index(polygon_ids, left)
            if left_idx is not None:
                polygon_to_polygon_edge_index.append(
                    torch.tensor([[lane_idx], [left_idx]], dtype=torch.long)
                )
                polygon_to_polygon_type.append(
                    torch.tensor(
                        [_polygon_to_polygon_types.index("LEFT")], dtype=torch.uint8
                    )
                )

        # Process right neighbors
        for right in lane.adj_lanes_right:
            right_idx = safe_list_index(polygon_ids, right)
            if right_idx is not None:
                polygon_to_polygon_edge_index.append(
                    torch.tensor([[lane_idx], [right_idx]], dtype=torch.long)
                )
                polygon_to_polygon_type.append(
                    torch.tensor(
                        [_polygon_to_polygon_types.index("RIGHT")], dtype=torch.uint8
                    )
                )

    # Concatenate edge indices and types
    if polygon_to_polygon_edge_index:
        polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
        polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
    else:
        polygon_to_polygon_edge_index = torch.empty((2, 0), dtype=torch.long)
        polygon_to_polygon_type = torch.empty(0, dtype=torch.uint8)

    map_data = {
        "map_polygon": {
            "num_nodes": len(all_polygons),
            "type": torch.tensor(polygon_types, dtype=torch.uint8),
            "light_type": torch.tensor(polygon_light_types, dtype=torch.uint8),
        },
        "map_point": {
            "num_nodes": sum(len(p) for p in point_positions),
            "position": torch.cat(point_positions, dim=0),
            "orientation": torch.cat(point_orientations, dim=0),
            "magnitude": torch.cat(point_magnitudes, dim=0),
            "type": torch.cat(point_types, dim=0),
        },
        ("map_point", "to", "map_polygon"): {"edge_index": point_to_polygon_edge_index},
        ("map_polygon", "to", "map_polygon"): {
            "edge_index": polygon_to_polygon_edge_index,
            "type": polygon_to_polygon_type,
        },
    }

    if dim > 2:
        map_data["map_point"]["height"] = torch.cat(point_heights, dim=0)

    return map_data


def safe_list_index(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return None


def convert_batch_to_smart(batch, output_dir: str):
    """Convert a single TrajData batch to SMART format and save it.

    Args:
        batch: TrajData batch to convert
        output_dir: Directory to save the converted data
    """
    # Process data
    agent_data = process_agent_batch(batch)
    map_data = process_map_features(batch)

    # Combine data
    data = {
        "scenario_id": batch.scene_ids[0],
        "city": "unknown",
        "agent": agent_data,
    }
    data.update(map_data)

    # Save with scenario_id as filename
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
        history_sec=(1.0, 1.0),  # 10 frames of history
        future_sec=(8.0, 8.0),  # 80 frames of future
        # only_predict=[AgentType.VEHICLE],
        state_format="x,y,z,xd,yd,s,c",
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
