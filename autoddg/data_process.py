from datetime import datetime

import datamart_profiler


def dataset_profiler(data_frame):
    metadata = datamart_profiler.process_dataset(data_frame)
    profile_summary = []

    # Iterate through each column in the metadata and summarize the details
    for column_meta in metadata["columns"]:
        column_summary = f"**{column_meta['name']}**: "

        # Structural type
        structural_type = column_meta.get("structural_type", "Unknown")
        column_summary += f"Data is of type {structural_type.split('/')[-1].lower()}. "

        # Number of distinct values (if applicable)
        if "num_distinct_values" in column_meta:
            num_distinct_values = column_meta["num_distinct_values"]
            column_summary += f"There are {num_distinct_values} unique values. "

        # Semantic types (simplified)
        # semantic_types = column_meta.get('semantic_types', [])
        # if semantic_types:
        #     semantic_summary = ', '.join([semantic_type.split('/')[-1].replace('_', ' ').lower() for semantic_type in semantic_types])
        #     column_summary += f"Semantic types include: {semantic_summary}. "

        # Handle coverage (if available)
        if "coverage" in column_meta:
            low = 0
            high = 0
            for i in range(len(column_meta["coverage"])):
                if column_meta["coverage"][i]["range"]["gte"] < low:
                    low = column_meta["coverage"][i]["range"]["gte"]
                if column_meta["coverage"][i]["range"]["lte"] > high:
                    high = column_meta["coverage"][i]["range"]["lte"]
            column_summary += f"Coverage spans from {low} to {high}. "

        # Append the summarized profile for this column
        profile_summary.append(column_summary)

    final_profile_summary = (
        "The key data profile information for this dataset includes:\n"
        + "\n".join(profile_summary)
    )

    semantic_summary = []
    # Temporal coverage (if available)
    if "temporal_coverage" in metadata:
        print("temporal_coverage")
        for temp_cov in metadata["temporal_coverage"]:
            col_names = ", ".join(temp_cov["column_names"])
            temporal_resolution = temp_cov["temporal_resolution"]
            range_values = [entry["range"]["gte"] for entry in temp_cov["ranges"]] + [
                entry["range"]["lte"] for entry in temp_cov["ranges"]
            ]
            min_value = datetime.fromtimestamp(min(range_values)).strftime("%Y-%m-%d")
            max_value = datetime.fromtimestamp(max(range_values)).strftime("%Y-%m-%d")
            date_range = f"from {min_value} to {max_value}"
            semantic_summary.append(
                f"**Temporal coverage** for columns {col_names} with resolution {temporal_resolution}, covering {date_range}."
            )

    # Spatial coverage (if available)
    if "spatial_coverage" in metadata:
        print("spatial_coverage")
        for spatial_cov in metadata["spatial_coverage"]:
            col_names = ", ".join(spatial_cov["column_names"])
            spatial_resolution = spatial_cov["type"]
            semantic_summary.append(
                f"**Spatial coverage** for columns {col_names}, with type {spatial_resolution}."
            )

    # Attribute keywords
    # if 'attribute_keywords' in metadata:
    #     keywords = ', '.join(metadata['attribute_keywords'])
    #     semantic_summary.append(f"**Attribute keywords**: {keywords}.")

    # Final summary as a human-readable text
    final_semantic_summary = "\n".join(semantic_summary)

    return final_profile_summary, final_semantic_summary
