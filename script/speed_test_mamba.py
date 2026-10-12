import sys
import os
import time
import argparse
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ===================== 永久修复 XSSBlock 报错 =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ultralytics.nn.uav_modules.block import XSSBlock  # noqa: F401
# ================================================================

from ultralytics import RTDETR
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.tasks import attempt_load_weights


def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


def seed_everything(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def should_profile_module(name, module, leaf_only=True, keywords=None, exclude_keywords=None):
    if name == "":
        return False

    if leaf_only and (not is_leaf_module(module)):
        return False

    text = f"{name} {module.__class__.__name__}".lower()

    if exclude_keywords:
        for k in exclude_keywords:
            if k and k.lower() in text:
                return False

    if keywords:
        return any(k.lower() in text for k in keywords)

    return True


class ModuleProfiler:
    def __init__(self, model, device, leaf_only=True, keywords=None, exclude_keywords=None):
        self.model = model
        self.device = device
        self.leaf_only = leaf_only
        self.keywords = keywords or []
        self.exclude_keywords = exclude_keywords or []

        self.handles = []
        self.stats = {}
        self.pending = defaultdict(list)  # name -> list[(start_event, end_event)] / cpu timestamps

        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if not should_profile_module(
                name=name,
                module=module,
                leaf_only=self.leaf_only,
                keywords=self.keywords,
                exclude_keywords=self.exclude_keywords
            ):
                continue

            self.stats[name] = {
                "class": module.__class__.__name__,
                "calls": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
                "min_ms": float("inf"),
            }

            self.handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._make_post_hook(name)))

    def _make_pre_hook(self, name):
        def pre_hook(module, inputs):
            if self.device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                self.pending[name].append([start_event, None])
            else:
                t0 = time.perf_counter()
                self.pending[name].append([t0, None])
        return pre_hook

    def _make_post_hook(self, name):
        def post_hook(module, inputs, outputs):
            if not self.pending[name]:
                return

            if self.device.type == "cuda":
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                self.pending[name][-1][1] = end_event
            else:
                t1 = time.perf_counter()
                self.pending[name][-1][1] = t1
        return post_hook

    def finalize_step(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        for name, items in self.pending.items():
            for start_obj, end_obj in items:
                if end_obj is None:
                    continue

                if self.device.type == "cuda":
                    ms = start_obj.elapsed_time(end_obj)
                else:
                    ms = (end_obj - start_obj) * 1000.0

                self.stats[name]["calls"] += 1
                self.stats[name]["total_ms"] += ms
                self.stats[name]["max_ms"] = max(self.stats[name]["max_ms"], ms)
                self.stats[name]["min_ms"] = min(self.stats[name]["min_ms"], ms)

        self.pending.clear()

    def remove(self):
        for h in self.handles:
            h.remove()

    def summary(self, avg_batch_ms, batch_size, topk=30):
        rows = []
        for name, info in self.stats.items():
            if info["calls"] == 0:
                continue
            avg_ms = info["total_ms"] / info["calls"]
            rows.append({
                "name": name,
                "class": info["class"],
                "calls": info["calls"],
                "total_ms": info["total_ms"],
                "avg_ms": avg_ms,
                "min_ms": 0.0 if info["min_ms"] == float("inf") else info["min_ms"],
                "max_ms": info["max_ms"],
                "pct_of_batch": (avg_ms / avg_batch_ms * 100.0) if avg_batch_ms > 0 else 0.0,
                "per_image_ms": avg_ms / batch_size,
            })

        rows.sort(key=lambda x: x["avg_ms"], reverse=True)
        return rows[:topk], rows


def prepare_model(weights, device):
    if weights.endswith(".pt"):
        model = attempt_load_weights(weights=weights, device=device)
        print(f"Loaded {weights}")
    else:
        model = RTDETR(weights).model

    model = model.to(device)
    model.eval()

    if hasattr(model, "fuse"):
        try:
            model.fuse()
        except Exception as e:
            print(f"[WARN] model.fuse() failed: {e}")

    return model


def cast_model(model, precision="fp32", channels_last=False):
    if precision == "fp16":
        model = model.half()
    else:
        model = model.float()

    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    return model


def prepare_input(batch, imgs, device, precision="fp32", channels_last=False):
    x = torch.randn((batch, 3, *imgs), device=device)

    if precision == "fp16":
        x = x.half()
    else:
        x = x.float()

    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    return x


def run_forward(model, x, precision="fp32"):
    if precision == "amp":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x)
    return model(x)


def benchmark(model, x, device, warmup=50, testtime=100, precision="fp32", profiler=None):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    print("begin warmup...")
    with torch.inference_mode():
        for _ in tqdm(range(warmup), desc="warmup"):
            _ = run_forward(model, x, precision=precision)
            if profiler is not None:
                profiler.finalize_step()

    print("begin test latency...")
    batch_times_ms = []

    with torch.inference_mode():
        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            for _ in tqdm(range(testtime), desc="test latency"):
                starter.record()
                _ = run_forward(model, x, precision=precision)
                ender.record()
                torch.cuda.synchronize()
                batch_times_ms.append(starter.elapsed_time(ender))
                if profiler is not None:
                    profiler.finalize_step()
        else:
            for _ in tqdm(range(testtime), desc="test latency"):
                t0 = time.perf_counter()
                _ = run_forward(model, x, precision=precision)
                t1 = time.perf_counter()
                batch_times_ms.append((t1 - t0) * 1000.0)
                if profiler is not None:
                    profiler.finalize_step()

    return np.array(batch_times_ms, dtype=np.float64)


def print_latency_summary(time_arr_ms, batch, weights):
    avg_batch_ms = float(np.mean(time_arr_ms))
    std_batch_ms = float(np.std(time_arr_ms))
    p50 = float(np.percentile(time_arr_ms, 50))
    p90 = float(np.percentile(time_arr_ms, 90))
    p95 = float(np.percentile(time_arr_ms, 95))

    per_image_ms = avg_batch_ms / batch
    fps = batch * 1000.0 / avg_batch_ms

    print()
    if weights.endswith(".pt"):
        print(f"model weights: {weights} size: {get_weight_size(weights)} MB")
    else:
        print(f"model yaml: {weights}")
    print(f"batch latency: {avg_batch_ms:.3f} ms ± {std_batch_ms:.3f} ms")
    print(f"per-image latency: {per_image_ms:.3f} ms")
    print(f"P50/P90/P95 batch latency: {p50:.3f}/{p90:.3f}/{p95:.3f} ms")
    print(f"FPS: {fps:.2f}")

    return avg_batch_ms, fps


def print_profile_table(rows, title):
    print(f"\n[{title}]")
    print("-" * 140)
    print(
        f'{"module name":55s} '
        f'{"class":24s} '
        f'{"avg(ms)":>10s} '
        f'{"per-img":>10s} '
        f'{"pct":>8s} '
        f'{"calls":>8s} '
        f'{"min":>10s} '
        f'{"max":>10s}'
    )
    print("-" * 140)

    for r in rows:
        print(
            f'{r["name"][:55]:55s} '
            f'{r["class"][:24]:24s} '
            f'{r["avg_ms"]:10.3f} '
            f'{r["per_image_ms"]:10.3f} '
            f'{r["pct_of_batch"]:8.2f} '
            f'{r["calls"]:8d} '
            f'{r["min_ms"]:10.3f} '
            f'{r["max_ms"]:10.3f}'
        )
    print("-" * 140)


def save_csv(rows, save_path):
    import csv
    keys = ["name", "class", "avg_ms", "per_image_ms", "pct_of_batch", "calls", "min_ms", "max_ms", "total_ms"]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nCSV saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="/root/autodl-tmp/UAV-DETR-main-33/best.pt")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgs", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--testtime", type=int, default=20)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "amp"])
    parser.add_argument("--channels-last", action="store_true")

    parser.add_argument("--profile", action="store_true", help="enable module profiling")
    parser.add_argument("--leaf-only", action="store_true", help="profile leaf modules only")
    parser.add_argument("--keywords", nargs="*", default=None,
                        help="only profile modules whose name/class contains these keywords")
    parser.add_argument("--exclude-keywords", nargs="*", default=["bn", "relu"],
                        help="exclude modules by keywords")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--save-csv", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)

    opt = parser.parse_args()
    seed_everything(opt.seed)

    device = select_device(opt.device, batch=opt.batch)
    print(f"Device: {device}")
    print(f"Precision: {opt.precision}")
    print(f"Batch: {opt.batch}, Image size: {opt.imgs}, channels_last: {opt.channels_last}")

    model = prepare_model(opt.weights, device)
    model = cast_model(model, precision=opt.precision, channels_last=opt.channels_last)
    x = prepare_input(opt.batch, opt.imgs, device, precision=opt.precision, channels_last=opt.channels_last)

    profiler = None
    if opt.profile:
        profiler = ModuleProfiler(
            model=model,
            device=device,
            leaf_only=opt.leaf_only,
            keywords=opt.keywords,
            exclude_keywords=opt.exclude_keywords
        )

    time_arr_ms = benchmark(
        model=model,
        x=x,
        device=device,
        warmup=opt.warmup,
        testtime=opt.testtime,
        precision=opt.precision,
        profiler=profiler
    )

    avg_batch_ms, fps = print_latency_summary(time_arr_ms, opt.batch, opt.weights)

    if profiler is not None:
        top_rows, all_rows = profiler.summary(avg_batch_ms=avg_batch_ms, batch_size=opt.batch, topk=opt.topk)
        title = "Module Profile Top-K"
        if opt.leaf_only:
            title += " (leaf only)"
        if opt.keywords:
            title += f" keywords={opt.keywords}"
        print_profile_table(top_rows, title)

        if opt.save_csv:
            save_csv(all_rows, opt.save_csv)

        profiler.remove()

#python speed_test_mamba.py   --weights /root/autodl-tmp/UAV-DETR-main-33/best.pt   --batch 4 --imgs 640 640 --device 0 --precision fp32   --profile --topk 30