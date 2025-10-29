import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import numpy as np

def compute_spatial_metrics(cfg: DictConfig):
    if cfg.compute_metrics.spatial is None:
        return
    cm_cfg = cfg.compute_metrics.spatial

    gt_filename: str = cm_cfg.gt_filename
    pred_root = Path(cm_cfg.pred_root)
    out_dir = Path(cm_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated_file = Path(cm_cfg.aggregated_output_filename)
    aggregated_file.parent.mkdir(parents=True, exist_ok=True)

    ks: list[int] = list(cm_cfg.l2_top_ks)
    layers_filter = {str(layer_idx) for layer_idx in cm_cfg.layers}

    methods = ["splat", "static_graph", "frame_attn", "splat_graph", "frame_attn_refine"]

    # Dataset-wide accumulators (layerwise):
    # methods -> class -> layer_key -> {sum: np.array[K], count: int}
    dataset_stats: dict[str, dict[str, dict[str, dict[str, np.ndarray | int]]]] = {
        m: {"objects": {}, "actions": {}, "all": {}} for m in methods
    }

    def _min_l2_at_k(pred_coords: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
        # pred_coords: [N, 2] (x, y); gt_xy: [2]
        if pred_coords.size == 0:
            return np.full(len(ks), np.inf, dtype=np.float64)
        diffs = pred_coords.astype(np.float64) - gt_xy[None, :]
        dists = np.sqrt((diffs[:, 0] ** 2) + (diffs[:, 1] ** 2))  # [N]
        out = np.empty(len(ks), dtype=np.float64)
        for i, k in enumerate(ks):
            kk = min(k, dists.shape[0])
            if kk <= 0:
                out[i] = np.inf
            else:
                out[i] = float(np.min(dists[:kk]))
        return out

    for clip in cfg.clips:
        clip_name = str(clip.name)
        gt_path = Path(cfg.preprocessed_root) / clip_name / gt_filename
        pred_path = pred_root / f"{clip_name}.json"

        if not gt_path.exists() or not pred_path.exists():
            continue

        with gt_path.open("r") as f:
            gt_data = json.load(f)
        with pred_path.open("r") as f:
            preds_all = json.load(f)

        per_clip_results = {}

        for method in methods:
            if method not in preds_all:
                continue

            # Per-class, per-layer accumulators for this clip
            sums: dict[str, dict[str, np.ndarray]] = {
                "objects": {},
                "actions": {},
                "all": {},
            }
            counts: dict[str, dict[str, int]] = {"objects": {}, "actions": {}, "all": {}}
            clip_method_items: list[dict] = []

            method_preds = preds_all[method]

            # Iterate timesteps present in both GT and predictions
            for t_key, gt_entry in gt_data.items():
                if t_key not in method_preds:
                    continue
                pred_entry = method_preds[t_key]

                for group in ("objects", "actions"):
                    gt_list = gt_entry.get(group, [])
                    pred_list = pred_entry.get(group, [])
                    n = min(len(gt_list), len(pred_list))
                    if n <= 0:
                        continue

                    for i in range(n):
                        gt_item = gt_list[i]
                        pred_item = pred_list[i]

                        # Ground-truth pixel point (x, y)
                        gx = float(gt_item["pixel_x"])  # assume set in config pipeline
                        gy = float(gt_item["pixel_y"])  # assume set in config pipeline
                        gt_xy = np.array([gx, gy], dtype=np.float64)

                        # Collect per-layer min_l2@k for this query
                        preds_by_layer = pred_item.get("predictions", {})
                        per_layer_out: dict[str, dict[str, float]] = {}
                        for layer_key, layer_pred in preds_by_layer.items():
                            lkey = str(layer_key)
                            if lkey not in layers_filter:
                                continue
                            coords = np.array(layer_pred.get("pixel_coords", []), dtype=np.float64)
                            vals = _min_l2_at_k(coords, gt_xy)
                            vals = np.round(vals, 2)
                            per_layer_out[lkey] = {f"min_l2@{k}": float(v) for k, v in zip(ks, vals.tolist())}
                            # accumulate per group/layer
                            if lkey not in sums[group]:
                                sums[group][lkey] = np.zeros(len(ks), dtype=np.float64)
                                counts[group][lkey] = 0
                            sums[group][lkey] += vals
                            counts[group][lkey] += 1
                            # overall
                            if lkey not in sums["all"]:
                                sums["all"][lkey] = np.zeros(len(ks), dtype=np.float64)
                                counts["all"][lkey] = 0
                            sums["all"][lkey] += vals
                            counts["all"][lkey] += 1

                        # record per-query item if we computed any layer metrics
                        if per_layer_out:
                            query_text = pred_item.get("query") or gt_item.get("query")
                            clip_method_items.append(
                                {
                                    "timestep": t_key,
                                    "frame_number": int(gt_entry.get("frame_number", -1)),
                                    "group": group,
                                    "query": query_text,
                                    "per_layer": per_layer_out,
                                }
                            )

            # Compute averages for this clip and method
            method_out = {"per_class": {}, "counts": {}, "items": clip_method_items}
            for group in ("objects", "actions", "all"):
                per_layer_avgs = {}
                per_layer_counts = {}
                for lkey, svec in sums[group].items():
                    c = counts[group].get(lkey, 0)
                    if c > 0:
                        avg_arr = svec / c
                        avg_list = np.round(avg_arr, 2).tolist()
                    else:
                        avg_list = [float("nan")] * len(ks)
                    per_layer_avgs[lkey] = {f"min_l2@{k}": v for k, v in zip(ks, avg_list)}
                    per_layer_counts[lkey] = c

                    # Update dataset-wide accumulators
                    if lkey not in dataset_stats[method][group]:
                        dataset_stats[method][group][lkey] = {
                            "sum": np.zeros(len(ks), dtype=np.float64),
                            "count": 0,
                        }
                    dataset_stats[method][group][lkey]["sum"] += svec
                    dataset_stats[method][group][lkey]["count"] += c

                method_out["per_class"][group] = per_layer_avgs
                method_out["counts"][group] = per_layer_counts

            per_clip_results[method] = method_out

        # Save per-clip file with query-wise metrics
        clip_out = {
            "clip": clip_name,
            "methods": per_clip_results,
        }
        with (out_dir / f"{clip_name}.json").open("w") as f:
            json.dump(clip_out, f, indent=2)

    # Save dataset-wide summary
    summary = {"methods": {}}
    for method in methods:
        if method not in dataset_stats:
            continue
        mstats = dataset_stats[method]
        out_m = {"per_class": {}, "counts": {}}
        for group in ("objects", "actions", "all"):
            per_layer = {}
            per_layer_counts = {}
            for lkey, stat in mstats[group].items():
                total_count = int(stat["count"])  # type: ignore[index]
                sums_arr = stat["sum"]  # type: ignore[index]
                if total_count > 0:
                    avg_arr = sums_arr / total_count  # type: ignore[operator]
                    avg_list = np.round(avg_arr, 2).tolist()
                else:
                    avg_list = [float("nan")] * len(ks)
                per_layer[lkey] = {f"min_l2@{k}": v for k, v in zip(ks, avg_list)}
                per_layer_counts[lkey] = total_count
            out_m["per_class"][group] = per_layer
            out_m["counts"][group] = per_layer_counts
        summary["methods"][method] = out_m

    with aggregated_file.open("w") as f:
        json.dump({"summary": summary}, f, indent=2)

def compute_temporal_metrics(cfg: DictConfig):
    if cfg.compute_metrics.temporal is None:
        return
    cm_cfg = cfg.compute_metrics.temporal

    pred_root = Path(cm_cfg.pred_root)
    labels_root = Path(cm_cfg.labels_root)
    labels_tmpl: str = cm_cfg.labels_filename_template
    out_dir = Path(cm_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated_file = Path(cm_cfg.aggregated_output_filename)
    aggregated_file.parent.mkdir(parents=True, exist_ok=True)

    # Metric params from eval config
    tcfg = cfg.eval.temporal.metrics

    # Dataset accumulators per ablation and query type
    dataset: dict[str, dict[str, list[dict]]] = {}

    # Helpers mirroring evaluator logic
    def _eval_frame_error(predicted: dict | None, gt: dict, tol: int) -> dict:
        if predicted is None or 'frame' not in predicted:
            return {
                'frame_error': float('inf'),
                'within_tolerance': False,
                'tolerance_used': tol,
                'success': False,
            }
        err = abs(int(predicted['frame']) - int(gt['frame']))
        return {
            'frame_error': err,
            'within_tolerance': err <= tol,
            'tolerance_used': tol,
            'success': err <= tol,
        }

    def _eval_iou(predicted: dict | None, gt: dict, thr: float) -> dict:
        if predicted is None or 'ranges' not in predicted:
            return {
                'iou': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'success': False,
                'threshold': thr,
            }
        def ranges_to_set(ranges):
            s = set()
            for a, b in ranges:
                s.update(range(int(a), int(b) + 1))
            return s
        gt_frames = ranges_to_set(gt['ranges'])
        pred_frames = ranges_to_set(predicted['ranges'])
        inter = len(gt_frames & pred_frames)
        union = len(gt_frames | pred_frames)
        iou = inter / union if union > 0 else 0.0
        prec = inter / len(pred_frames) if len(pred_frames) > 0 else 0.0
        rec = inter / len(gt_frames) if len(gt_frames) > 0 else 0.0
        return {
            'iou': iou,
            'precision': prec,
            'recall': rec,
            'success': iou >= thr,
            'threshold': thr,
        }

    def _eval_ordering(predicted: dict | None, gt: dict, order_weight: float, iou_weight: float) -> dict:
        if predicted is None or 'events' not in predicted:
            return {
                'order_correct': False,
                'per_event_iou': [],
                'mean_iou': 0.0,
                'composite_score': 0.0,
                'success': False,
            }
        pred_events = predicted['events']
        gt_events = gt['events']
        order_correct = len(pred_events) == len(gt_events)
        if order_correct:
            for i in range(len(pred_events)):
                if int(pred_events[i].get('order', -1)) != int(gt_events[i]['order']):
                    order_correct = False
                    break
        per_event_iou: list[float] = []
        for i in range(min(len(pred_events), len(gt_events))):
            pr = pred_events[i]['frame_range']
            gr = gt_events[i]['frame_range']
            pf = set(range(int(pr[0]), int(pr[1]) + 1))
            gf = set(range(int(gr[0]), int(gr[1]) + 1))
            inter = len(pf & gf)
            union = len(pf | gf)
            per_event_iou.append(inter / union if union > 0 else 0.0)
        mean_iou = float(np.mean(per_event_iou)) if per_event_iou else 0.0
        composite = order_weight * (1.0 if order_correct else 0.0) + iou_weight * mean_iou
        return {
            'order_correct': order_correct,
            'per_event_iou': per_event_iou,
            'mean_iou': mean_iou,
            'composite_score': composite,
            'success': composite >= 0.5,
        }

    def _eval_count(predicted: dict | None, gt: dict, count_weight: float, iou_weight: float) -> dict:
        if predicted is None or 'count' not in predicted:
            return {
                'count_correct': False,
                'count_error': int(gt['count']),
                'per_occurrence_iou': [],
                'mean_iou': 0.0,
                'composite_score': 0.0,
                'success': False,
            }
        pred_count = int(predicted['count'])
        gt_count = int(gt['count'])
        pred_occs = predicted.get('occurrences', []) or []
        gt_occs = gt.get('occurrences', []) or []
        count_correct = (pred_count == gt_count)
        count_error = abs(pred_count - gt_count)
        per_iou: list[float] = []
        matched = set()
        for po in pred_occs[:len(gt_occs)]:
            pf = set(range(int(po[0]), int(po[1]) + 1))
            best_iou = 0.0
            best_idx = -1
            for j, go in enumerate(gt_occs):
                if j in matched:
                    continue
                gf = set(range(int(go[0]), int(go[1]) + 1))
                inter = len(pf & gf)
                union = len(pf | gf)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_idx >= 0:
                matched.add(best_idx)
                per_iou.append(best_iou)
        mean_iou = float(np.mean(per_iou)) if per_iou else 0.0
        composite = count_weight * (1.0 if count_correct else 0.0) + iou_weight * mean_iou
        return {
            'count_correct': count_correct,
            'count_error': count_error,
            'pred_count': pred_count,
            'gt_count': gt_count,
            'per_occurrence_iou': per_iou,
            'mean_iou': mean_iou,
            'composite_score': composite,
            'success': composite >= 0.5,
        }

    # Per-clip processing
    for clip in cfg.clips:
        clip_name = str(clip.name)
        pred_path = pred_root / f"{clip_name}.json"
        gt_path = labels_root / labels_tmpl.format(clip_name=clip_name)
        if not pred_path.exists() or not gt_path.exists():
            continue

        with pred_path.open('r') as f:
            preds = json.load(f)
        with gt_path.open('r') as f:
            gt_full = json.load(f)
        gt_by_id = {a['query_id']: a for a in gt_full.get('annotations', [])}

        per_clip_out: dict[str, dict] = {}

        for ablation, items in preds.get('ablations', {}).items():
            results = []
            for item in items:
                qid = item.get('query_id')
                qtype = item.get('query_type')
                pred = item.get('predicted')
                gt_entry = gt_by_id.get(qid, {})
                gt = gt_entry.get('ground_truth')
                metrics = {}
                if gt is not None:
                    if qtype in ('action_onset', 'action_offset'):
                        tol = int(tcfg[qtype]['tolerance'])
                        metrics = _eval_frame_error(pred, gt, tol)
                    elif qtype == 'action_duration':
                        thr = float(tcfg[qtype]['threshold'])
                        metrics = _eval_iou(pred, gt, thr)
                    elif qtype == 'multiple_event_ordering':
                        ow = float(tcfg[qtype]['order_weight'])
                        iw = float(tcfg[qtype]['iou_weight'])
                        metrics = _eval_ordering(pred, gt, ow, iw)
                    elif qtype == 'count_frequency':
                        cw = float(tcfg[qtype]['count_weight'])
                        iw = float(tcfg[qtype]['iou_weight'])
                        metrics = _eval_count(pred, gt, cw, iw)
                results.append({
                    'query_id': qid,
                    'query_type': qtype,
                    'question': item.get('question'),
                    'predicted': pred,
                    'ground_truth': gt,
                    'metrics': metrics,
                    'raw_response': item.get('raw_response'),
                })

            # Aggregate per ablation for this clip
            by_type: dict[str, list[dict]] = {}
            for r in results:
                by_type.setdefault(r['query_type'], []).append(r)

            aggregated: dict[str, dict] = {}
            # action_onset/offset
            for qtype in ('action_onset', 'action_offset'):
                qres = by_type.get(qtype, [])
                if qres:
                    errors = [m['metrics']['frame_error'] for m in qres if m['metrics'].get('frame_error') != float('inf')]
                    success_rate = sum(1 for m in qres if m['metrics'].get('success')) / len(qres)
                    aggregated[qtype] = {
                        'mean_frame_error': float(np.mean(errors)) if errors else float('inf'),
                        'success_rate': success_rate,
                        'count': len(qres),
                    }
            # action_duration
            qres = by_type.get('action_duration', [])
            if qres:
                ious = [m['metrics'].get('iou', 0.0) for m in qres]
                success_rate = sum(1 for m in qres if m['metrics'].get('success')) / len(qres)
                aggregated['action_duration'] = {
                    'mean_iou': float(np.mean(ious)) if ious else 0.0,
                    'success_rate': success_rate,
                    'count': len(qres),
                }
            # multiple_event_ordering
            qres = by_type.get('multiple_event_ordering', [])
            if qres:
                scores = [m['metrics'].get('composite_score', 0.0) for m in qres]
                order_correct_rate = sum(1 for m in qres if m['metrics'].get('order_correct')) / len(qres)
                mean_ious = [m['metrics'].get('mean_iou', 0.0) for m in qres]
                aggregated['multiple_event_ordering'] = {
                    'mean_composite_score': float(np.mean(scores)) if scores else 0.0,
                    'order_correct_rate': order_correct_rate,
                    'mean_iou': float(np.mean(mean_ious)) if mean_ious else 0.0,
                    'count': len(qres),
                }
            # count_frequency
            qres = by_type.get('count_frequency', [])
            if qres:
                scores = [m['metrics'].get('composite_score', 0.0) for m in qres]
                count_correct_rate = sum(1 for m in qres if m['metrics'].get('count_correct')) / len(qres)
                mean_ious = [m['metrics'].get('mean_iou', 0.0) for m in qres]
                aggregated['count_frequency'] = {
                    'mean_composite_score': float(np.mean(scores)) if scores else 0.0,
                    'count_correct_rate': count_correct_rate,
                    'mean_iou': float(np.mean(mean_ious)) if mean_ious else 0.0,
                    'count': len(qres),
                }

            per_clip_out[ablation] = {
                'metrics': aggregated,
                'results': results,
            }

            # Add to dataset accumulators
            dataset.setdefault(ablation, {}).setdefault('items', []).extend(results)

        # Save per-clip results file
        with (out_dir / f"{clip_name}.json").open('w') as f:
            json.dump({'clip': clip_name, 'ablations': per_clip_out}, f, indent=2)

    # Aggregate dataset-wide per ablation
    summary: dict[str, dict] = {}
    for ablation, data in dataset.items():
        items = data.get('items', [])
        by_type: dict[str, list[dict]] = {}
        for r in items:
            by_type.setdefault(r['query_type'], []).append(r)
        agg: dict[str, dict] = {}
        for qtype in ('action_onset', 'action_offset'):
            qres = by_type.get(qtype, [])
            if qres:
                errors = [m['metrics']['frame_error'] for m in qres if m['metrics'].get('frame_error') != float('inf')]
                success_rate = sum(1 for m in qres if m['metrics'].get('success')) / len(qres)
                agg[qtype] = {
                    'mean_frame_error': float(np.mean(errors)) if errors else float('inf'),
                    'success_rate': success_rate,
                    'count': len(qres),
                }
        qres = by_type.get('action_duration', [])
        if qres:
            ious = [m['metrics'].get('iou', 0.0) for m in qres]
            success_rate = sum(1 for m in qres if m['metrics'].get('success')) / len(qres)
            agg['action_duration'] = {
                'mean_iou': float(np.mean(ious)) if ious else 0.0,
                'success_rate': success_rate,
                'count': len(qres),
            }
        qres = by_type.get('multiple_event_ordering', [])
        if qres:
            scores = [m['metrics'].get('composite_score', 0.0) for m in qres]
            order_correct_rate = sum(1 for m in qres if m['metrics'].get('order_correct')) / len(qres)
            mean_ious = [m['metrics'].get('mean_iou', 0.0) for m in qres]
            agg['multiple_event_ordering'] = {
                'mean_composite_score': float(np.mean(scores)) if scores else 0.0,
                'order_correct_rate': order_correct_rate,
                'mean_iou': float(np.mean(mean_ious)) if mean_ious else 0.0,
                'count': len(qres),
            }
        qres = by_type.get('count_frequency', [])
        if qres:
            scores = [m['metrics'].get('composite_score', 0.0) for m in qres]
            count_correct_rate = sum(1 for m in qres if m['metrics'].get('count_correct')) / len(qres)
            mean_ious = [m['metrics'].get('mean_iou', 0.0) for m in qres]
            agg['count_frequency'] = {
                'mean_composite_score': float(np.mean(scores)) if scores else 0.0,
                'count_correct_rate': count_correct_rate,
                'mean_iou': float(np.mean(mean_ious)) if mean_ious else 0.0,
                'count': len(qres),
            }
        summary[ablation] = {'metrics': agg}

    with aggregated_file.open('w') as f:
        json.dump({'ablations': summary}, f, indent=2)

def compute_triplets_metrics(cfg: DictConfig):
    if cfg.compute_metrics.triplets is None:
        return

@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    compute_spatial_metrics(cfg)
    compute_temporal_metrics(cfg)
    compute_triplets_metrics(cfg)

if __name__ == "__main__":
    main()