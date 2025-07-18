#!/usr/bin/env python3
"""
Example script demonstrating how to use the metadata filling functionality
for IPBES positive datasets.

This script shows how to:
1. Load positive datasets
2. Identify instances with missing metadata
3. Fill missing metadata using CrossRef API with rate limiting
4. Save modified instances to separate files
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_pipeline.ipbes.create_ipbes_raw import get_ipbes_positives, rename_positives
from src.data_pipeline.ipbes.fetch import fill_missing_metadata, identify_missing_metadata
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    Main function demonstrating metadata filling functionality.
    """
    logger.info("Starting metadata filling example...")
    
    # Step 1: Load positive datasets
    logger.info("Loading positive datasets...")
    try:
        pos_ds_list = get_ipbes_positives()
        logger.info(f"Loaded {len(pos_ds_list)} positive datasets")
        
        # Rename columns for consistency
        pos_ds_list = [rename_positives(ds) for ds in pos_ds_list]
        
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return
    
    # Step 2: Process each dataset
    data_type_names = ["IAS", "SUA", "VA"]
    output_dir = "examples/output/modified_instances"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, pos_ds in enumerate(pos_ds_list):
        data_type = data_type_names[i] if i < len(data_type_names) else f"dataset_{i}"
        logger.info(f"\n--- Processing {data_type} dataset ---")
        logger.info(f"Dataset size: {len(pos_ds)} instances")
        
        # Step 3: Identify instances with missing metadata
        missing_instances = identify_missing_metadata(pos_ds)
        logger.info(f"Found {len(missing_instances)} instances with missing title/abstract")
        
        if not missing_instances:
            logger.info(f"No missing metadata found in {data_type} dataset")
            continue
        
        # Step 4: Fill missing metadata (limit to first 10 for demo purposes)
        demo_limit = min(10, len(missing_instances))
        logger.info(f"Demo: Processing first {demo_limit} instances with missing metadata...")
        
        # Create a smaller dataset for demo
        demo_indices = [idx for idx, _ in missing_instances[:demo_limit]]
        demo_dataset = pos_ds.select(demo_indices)
        
        # Fill metadata for demo dataset
        output_file = os.path.join(output_dir, f"{data_type}_demo_modified_instances.csv")
        
        try:
            updated_dataset, modified_instances = fill_missing_metadata(
                demo_dataset,
                output_file=output_file,
                max_workers=3  # Conservative for demo
            )
            
            logger.info(f"Successfully updated {len(modified_instances)} instances")
            
            # Show some statistics
            if modified_instances:
                title_updates = sum(1 for inst in modified_instances if 'title' in inst['updated_fields'])
                abstract_updates = sum(1 for inst in modified_instances if 'abstract' in inst['updated_fields'])
                logger.info(f"  - Title updates: {title_updates}")
                logger.info(f"  - Abstract updates: {abstract_updates}")
                logger.info(f"  - Modified instances saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {data_type} dataset: {e}")
            continue
    
    logger.info("\nMetadata filling example completed!")
    logger.info(f"Check the output directory: {output_dir}")


if __name__ == "__main__":
    main() 