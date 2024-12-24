from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentType, SceneBatch, UnifiedDataset
from trajdata.augmentation import NoiseHistories
from trajdata.visualization.vis import plot_scene_batch


def main():
    noise_hists = NoiseHistories()

    # dataset = UnifiedDataset(
    #     desired_data=["nuplan_mini"],
    #     centric="scene",
    #     desired_dt=0.1,
    #     history_sec=(3.2, 3.2),
    #     future_sec=(4.8, 4.8),
    #     # only_types=[AgentType.VEHICLE],
    #     # agent_interaction_distances=defaultdict(lambda: 30.0),
    #     # incl_robot_future=True,
    #     # incl_raster_map=True,
    #     # raster_map_params={
    #     #     "px_per_m": 2,
    #     #     "map_size_px": 224,
    #     #     "offset_frac_xy": (-0.5, 0.0),
    #     # },
    #     # augmentations=[noise_hists],
    #     max_agent_num=20,
    #     num_workers=4,
    #     verbose=True,
    #     data_dirs={  # Remember to change this to match your filesystem!
    #         "nuplan_mini": "nuplan/dataset/nuplan-v1.1",
    #     },
    # )
    dataset = UnifiedDataset(
        desired_data=["nuplan_mini"],
        centric="scene",
        desired_dt=0.1,
        # history_sec=(-float("inf"), None),
        # future_sec=(0, 0),
        only_predict=[AgentType.VEHICLE],
        state_format="x,y,z,xd,yd,h",
        obs_format="x,y,z,xd,yd,s,c",
        # agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=False,
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 1,
            "map_size_px": 100,
            # "offset_frac_xy": (-0.5, 0.0),
        },
        num_workers=0,
        verbose=True,
        incl_vector_map=True,
        vector_map_params={
            "incl_road_lanes": True,
            "incl_road_areas": True,
            "incl_ped_crosswalks": True,
            "incl_ped_walkways": True,
        },
        data_dirs={  # Remember to change this to match your filesystem!
            # "waymo_val": "/home/haoweis/trajdata_smart/trajdata/data/waymo/",
            "nuplan_mini": "nuplan/dataset/nuplan-v1.1",
        },
    )


    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        num_workers=4,
    )

    batch: SceneBatch
    for batch in tqdm(dataloader):
        print(batch.scene_ids)
        plot_scene_batch(batch, batch_idx=0, plot_vec_map=True)


if __name__ == "__main__":
    main()
