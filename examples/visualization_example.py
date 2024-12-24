from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.visualization.interactive_animation import (
    InteractiveAnimation,
    animate_agent_batch_interactive,
)
from trajdata.visualization.interactive_vis import plot_agent_batch_interactive
from trajdata.visualization.vis import plot_agent_batch


def main():
    dataset = UnifiedDataset(
        desired_data=["nuplan_mini"],
        centric="agent",
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
            "offset_frac_xy": (-0.5, 0.0),
        },
        num_workers=4,
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
        num_workers=0,
    )

    batch: AgentBatch
    for batch in tqdm(dataloader):
        print(batch.scene_ids)
        # print(batch.agent_name)
        plot_agent_batch_interactive(batch, batch_idx=0, cache_path=dataset.cache_path)
        plot_agent_batch(batch, batch_idx=0)

        animation = InteractiveAnimation(
            animate_agent_batch_interactive,
            batch=batch,
            batch_idx=0,
            cache_path=dataset.cache_path,
        )
        animation.show()
        # break


if __name__ == "__main__":
    main()
