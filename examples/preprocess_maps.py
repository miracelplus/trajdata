from trajdata import UnifiedDataset


# @profile
def main():
    dataset = UnifiedDataset(
        # TODO(bivanovic@nvidia.com) Remove lyft from default examples
        desired_data=["nuplan_mini"],
        desired_dt=0.1,
        rebuild_maps=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "nuplan_mini": "/home/haoweis/trajdata_smart/trajdata/data/nuplan/dataset/nuplan-v1.1/",
        },
        verbose=True,
    )
    print(f"Finished Caching Maps!")


if __name__ == "__main__":
    main()
