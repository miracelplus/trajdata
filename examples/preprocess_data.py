import os

from trajdata import UnifiedDataset


def main():
    dataset = UnifiedDataset(
        desired_data=["nuplan_mini"],
        desired_dt=0.1,
        rebuild_cache=True,
        rebuild_maps=True,
        num_workers=os.cpu_count(),
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "nuplan_mini": "/home/haoweis/trajdata_smart/trajdata/data/nuplan/dataset/nuplan-v1.1/",
        },
    )
    print(f"Total Data Samples: {len(dataset):,}")


if __name__ == "__main__":
    main()
