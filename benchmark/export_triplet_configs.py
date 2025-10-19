#!/usr/bin/env python3
"""
Export triplet configurations from CholecT50 labels to CSV format.

This script analyzes CholecT50 annotation JSONs and exports a table with:
- Video ID
- Configuration ID  
- Start frame
- End frame
- Frame count
- Triplet IDs (valid actions only, excluding null/no-action)
- Triplet names
- Phase

Usage:
    python export_triplet_configs.py --video_ids 1 2 3 --output triplet_configs.csv
    python export_triplet_configs.py --all --output triplet_configs.csv
"""

import sys
import argparse
import csv
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.cholect50_utils import CholecT50Loader


def analyze_video_configs(
    video_id: int, 
    loader: CholecT50Loader,
    exclude_nulls: bool = True,
    exclude_no_action: bool = True,
    min_length: int = 1
) -> List[Dict]:
    """
    Analyze a video and extract all triplet configurations.
    
    Args:
        video_id: Video number
        loader: CholecT50Loader instance
        exclude_nulls: If True, exclude null triplets (IDs 94-99, instrument present but no action)
        exclude_no_action: If True, exclude no-action frames (ID -1)
        min_length: Minimum number of consecutive frames for a configuration
        
    Returns:
        List of configuration dictionaries
    """
    data = loader.load_video_annotations(video_id)
    
    current_config = None
    config_start = None
    configs = []
    prev_details = []
    
    for frame_num in sorted([int(k) for k in data['annotations'].keys()]):
        frame_data = data['annotations'][str(frame_num)]
        
        # Get triplet IDs for this frame
        triplet_ids = []
        triplet_details = []
        
        for triplet_data in frame_data:
            triplet_id = triplet_data[0]
            
            # Filter based on exclusion settings
            if exclude_no_action and triplet_id == -1:
                continue
            if exclude_nulls and triplet_id >= 94:
                continue
                
            triplet_ids.append(triplet_id)
            
            # Get details for valid triplets
            if 0 <= triplet_id < 94:
                instrument_id = triplet_data[1]
                verb_id = triplet_data[2]
                target_id = triplet_data[7]
                phase_id = triplet_data[14]
                
                triplet_details.append({
                    'id': triplet_id,
                    'name': data['categories']['triplet'][str(triplet_id)],
                    'instrument': data['categories']['instrument'][str(instrument_id)],
                    'verb': data['categories']['verb'][str(verb_id)],
                    'target': data['categories']['target'][str(target_id)],
                    'phase': data['categories']['phase'][str(phase_id)]
                })
        
        # Skip frames with no valid triplets if excluding
        if not triplet_ids and (exclude_nulls or exclude_no_action):
            if current_config is not None:
                configs.append({
                    'config_ids': current_config,
                    'details': prev_details,
                    'start': config_start,
                    'end': frame_num - 1,
                    'length': frame_num - config_start
                })
                current_config = None
            continue
        
        # Create configuration signature
        config_sig = tuple(sorted(triplet_ids))
        
        if config_sig != current_config:
            if current_config is not None:
                configs.append({
                    'config_ids': current_config,
                    'details': prev_details,
                    'start': config_start,
                    'end': frame_num - 1,
                    'length': frame_num - config_start
                })
            current_config = config_sig
            config_start = frame_num
            prev_details = triplet_details
    
    # Close last config
    if current_config is not None:
        last_frame = max([int(k) for k in data['annotations'].keys()])
        configs.append({
            'config_ids': current_config,
            'details': prev_details,
            'start': config_start,
            'end': last_frame,
            'length': last_frame - config_start + 1
        })
    
    # Filter by minimum length
    configs = [c for c in configs if c['length'] >= min_length]
    
    # Add video_id to each config
    for cfg in configs:
        cfg['video_id'] = video_id
    
    return configs


def export_to_csv(configs: List[Dict], output_path: Path):
    """Export configurations to CSV file."""
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = [
            'video_id',
            'config_id',
            'start_frame',
            'end_frame',
            'num_frames',
            'triplet_ids',
            'num_triplets',
            'triplet_names',
            'instruments',
            'verbs',
            'targets',
            'phase'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for cfg in configs:
            # Format triplet information
            triplet_ids = ','.join(map(str, cfg['config_ids']))
            triplet_names = ' | '.join([d['name'] for d in cfg['details']])
            instruments = ','.join([d['instrument'] for d in cfg['details']])
            verbs = ','.join([d['verb'] for d in cfg['details']])
            targets = ','.join([d['target'] for d in cfg['details']])
            phase = cfg['details'][0]['phase'] if cfg['details'] else ''
            
            writer.writerow({
                'video_id': cfg['video_id'],
                'config_id': cfg.get('config_id', ''),
                'start_frame': cfg['start'],
                'end_frame': cfg['end'],
                'num_frames': cfg['length'],
                'triplet_ids': triplet_ids,
                'num_triplets': len(cfg['details']),
                'triplet_names': triplet_names,
                'instruments': instruments,
                'verbs': verbs,
                'targets': targets,
                'phase': phase
            })


def main():
    parser = argparse.ArgumentParser(
        description='Export CholecT50 triplet configurations to CSV'
    )
    parser.add_argument(
        '--video_ids',
        type=int,
        nargs='+',
        help='Video IDs to process (e.g., 1 2 3)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all available videos'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='triplet_configs.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--min_length',
        type=int,
        default=10,
        help='Minimum number of frames for a configuration (default: 10)'
    )
    parser.add_argument(
        '--include_nulls',
        action='store_true',
        help='Include null triplets (instrument present, no action)'
    )
    parser.add_argument(
        '--include_no_action',
        action='store_true',
        help='Include no-action frames'
    )
    
    args = parser.parse_args()
    
    if args.all:
        video_ids = list(range(1, 51))  # Videos 1-50
    elif args.video_ids:
        video_ids = args.video_ids
    else:
        print("ERROR: Must specify either --video_ids or --all")
        return 1
    
    loader = CholecT50Loader()
    all_configs = []
    config_counter = 0
    
    print(f"Processing {len(video_ids)} videos...")
    print("Settings:")
    print(f"  - Min length: {args.min_length} frames")
    print(f"  - Include nulls: {args.include_nulls}")
    print(f"  - Include no-action: {args.include_no_action}")
    print()
    
    for video_id in video_ids:
        try:
            configs = analyze_video_configs(
                video_id,
                loader,
                exclude_nulls=not args.include_nulls,
                exclude_no_action=not args.include_no_action,
                min_length=args.min_length
            )
            
            # Add sequential config IDs
            for cfg in configs:
                config_counter += 1
                cfg['config_id'] = f"V{video_id:02d}_C{config_counter:04d}"
            
            all_configs.extend(configs)
            print(f"✓ Video {video_id:02d}: {len(configs)} configurations")
            
        except FileNotFoundError:
            print(f"✗ Video {video_id:02d}: Labels not found (skipping)")
        except Exception as e:
            print(f"✗ Video {video_id:02d}: Error - {e}")
    
    # Export to CSV
    output_path = Path(args.output)
    export_to_csv(all_configs, output_path)
    
    print()
    print(f"✓ Exported {len(all_configs)} configurations to {output_path}")
    print()
    print("Summary:")
    print(f"  - Total configurations: {len(all_configs)}")
    print(f"  - Single-triplet configs: {sum(1 for c in all_configs if len(c['details']) == 1)}")
    print(f"  - Multi-triplet configs: {sum(1 for c in all_configs if len(c['details']) >= 2)}")
    print(f"  - Average length: {sum(c['length'] for c in all_configs) / len(all_configs):.1f} frames")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

