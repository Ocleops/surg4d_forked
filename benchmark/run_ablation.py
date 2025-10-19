#!/usr/bin/env python3
"""
Main script to run ablation study for multi-frame triplet recognition.

Three conditions:
1. Single Frame: Baseline using middle frame
2. Multi-Frame: Temporal reasoning with multiple frames
3. Multi-Frame + Graph: Spatiotemporal reasoning with scene graph

Usage:
    python run_ablation.py --num_sequences 5 --frames_per_sequence 5
    python run_ablation.py --conditions single_frame multiframe
"""

import argparse
from pathlib import Path
from datetime import datetime

from benchmark_config import BenchmarkConfig
from multiframe_selector import MultiFrameSelector
from multiframe_evaluator import MultiFrameEvaluator


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation study for multi-frame triplet recognition'
    )
    parser.add_argument(
        '--num_sequences',
        type=int,
        default=5,
        help='Number of sequences to evaluate (default: 5)'
    )
    parser.add_argument(
        '--frames_per_sequence',
        type=int,
        default=5,
        help='Number of frames per sequence (default: 5)'
    )
    parser.add_argument(
        '--conditions',
        nargs='+',
        choices=['single_frame', 'multiframe', 'multiframe_graph'],
        default=['single_frame', 'multiframe', 'multiframe_graph'],
        help='Conditions to evaluate (default: all three)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: auto-generated in results/)'
    )
    parser.add_argument(
        '--use_4bit',
        action='store_true',
        help='Use 4-bit quantization for model'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = BenchmarkConfig(
        num_test_frames=args.num_sequences,
        model_name="qwen",
        qwen_version="qwen2.5",
        use_4bit_quantization=args.use_4bit,
        use_scene_graph=('multiframe_graph' in args.conditions),
        exact_match=True,
        save_responses=True,
        save_prompts=True,
        verbose=args.verbose
    )
    
    print("="*80)
    print("MULTI-FRAME ABLATION STUDY")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Number of sequences: {args.num_sequences}")
    print(f"  Frames per sequence: {args.frames_per_sequence}")
    print(f"  Conditions: {', '.join(args.conditions)}")
    print(f"  Model: Qwen2.5-VL-7B")
    print(f"  4-bit quantization: {args.use_4bit}")
    print()
    
    # Select samples
    print("Selecting multi-frame sequences...")
    selector = MultiFrameSelector(config)
    samples = selector.select_sequences(
        num_sequences=args.num_sequences,
        frames_per_sequence=args.frames_per_sequence,
        min_config_length=20
    )
    
    if not samples:
        print("ERROR: No samples selected! Check if preprocessed data is available.")
        return 1
    
    selector.print_summary(samples)
    
    # Run ablation study
    print("\nInitializing evaluator...")
    evaluator = MultiFrameEvaluator(config)
    
    results = evaluator.run_ablation_study(samples, conditions=args.conditions)
    
    # Print comparison
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print()
    print(f"{'Condition':<25} {'Instrument':<12} {'Verb':<12} {'Target':<12} {'Full Triplet':<12}")
    print("-"*80)
    
    for condition, data in results['conditions'].items():
        metrics = data['metrics']
        print(f"{condition:<25} "
              f"{metrics['instrument_acc']:>10.1%}  "
              f"{metrics['verb_acc']:>10.1%}  "
              f"{metrics['target_acc']:>10.1%}  "
              f"{metrics['triplet_acc']:>10.1%}")
    
    print("="*80)
    
    # Compute improvements
    if 'single_frame' in results['conditions'] and 'multiframe' in results['conditions']:
        single = results['conditions']['single_frame']['metrics']['triplet_acc']
        multi = results['conditions']['multiframe']['metrics']['triplet_acc']
        improvement = (multi - single) / single * 100 if single > 0 else 0
        print(f"\nTemporal improvement: {improvement:+.1f}%")
    
    if 'multiframe' in results['conditions'] and 'multiframe_graph' in results['conditions']:
        multi = results['conditions']['multiframe']['metrics']['triplet_acc']
        graph = results['conditions']['multiframe_graph']['metrics']['triplet_acc']
        improvement = (graph - multi) / multi * 100 if multi > 0 else 0
        print(f"Graph improvement: {improvement:+.1f}%")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config.results_dir / f"ablation_{timestamp}.json"
    
    evaluator.save_results(results, output_path)
    
    print(f"\n✓ Ablation study complete!")
    print(f"✓ Results saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

