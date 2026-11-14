import argparse
import collections
from collections import OrderedDict
import copy
import csv
import hashlib
import json
import os
import re
import uuid
import warnings
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from torch.ao import quantization as tq
from torchvision import datasets, models, transforms


HEAD = 'classifier.1._packed_params._packed_params'


def make_native_model(baseline_state):
    torch.backends.quantized.engine = 'qnnpack'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = models.quantization.mobilenet_v2(weights=None, num_classes=2).eval()
        model.qconfig = tq.get_default_qconfig('qnnpack')
        tq.prepare(model, inplace=True)
        tq.convert(model, inplace=True)
    model.load_state_dict(baseline_state, strict=True)
    return model.eval()


def clone_state(sd):
    out = OrderedDict()
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        elif isinstance(v, tuple):
            out[k] = tuple((x.clone() if torch.is_tensor(x) else x for x in v))
        else:
            out[k] = copy.deepcopy(v)
    if hasattr(sd, '_metadata'):
        out._metadata = copy.deepcopy(sd._metadata)
    return out


def logical_tensor(sd, key):
    if key == 'classifier.1.weight':
        return sd[HEAD][0]
    if key == 'classifier.1.bias':
        return sd[HEAD][1]
    return sd[key]


def set_tensor(sd, key, value):
    if key == 'classifier.1.weight':
        sd[HEAD] = (value, sd[HEAD][1])
    elif key == 'classifier.1.bias':
        sd[HEAD] = (sd[HEAD][0], value)
    else:
        sd[key] = value


def logical_template(sd):
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v) and (k.endswith('.weight') or k.endswith('.bias') or 'running_mean' in k or ('running_var' in k) or ('num_batches_tracked' in k)):
            out[k] = v
    out['classifier.1.weight'] = sd[HEAD][0]
    out['classifier.1.bias'] = sd[HEAD][1]
    return out


def bn_state_keys(model):
    keys = []
    for name, module in model.named_modules():
        if isinstance(module, torch.ao.nn.quantized.BatchNorm2d):
            keys.extend((name + '.' + k for k in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']))
    assert keys
    return keys


def apply_int8_update(state, donor, mask, bn_keys):
    new = clone_state(state)
    bn = set(bn_keys)
    audit = []
    for key, index in mask.items():
        if key in bn:
            continue
        old = logical_tensor(new, key)
        src = logical_tensor(donor, key)
        if not torch.is_tensor(src):
            raise ValueError(f'Update source is not a tensor: {key}')
        if old.is_quantized:
            if old.qscheme() != torch.per_tensor_affine or old.dtype != torch.qint8:
                raise ValueError('Expected the saved baseline qint8 per-tensor weight grid')
            scale = old.q_scale()
            zp = old.q_zero_point()
            values = (src.dequantize() if src.is_quantized else src).contiguous().view(-1)[index]
            lo = (-128 - zp) * scale
            hi = (127 - zp) * scale
            encoded = torch.quantize_per_tensor(values, scale, zp, torch.qint8).int_repr()
            ints = old.int_repr().contiguous()
            ints.view(-1)[index] = encoded
            updated = torch._make_per_tensor_quantized_tensor(ints, scale, zp)
            set_tensor(new, key, updated)
            audit.append(dict(tensor=key, transmitted_values=index.numel(), clipped_low=int((values < lo).sum()), clipped_high=int((values > hi).sum()), receiver_scale=scale, receiver_zero_point=zp))
        else:
            if src.is_quantized:
                raise ValueError(f'Unexpected quantized float/bias field {key}')
            updated = old.clone().contiguous()
            updated.view(-1)[index] = src.contiguous().view(-1)[index]
            set_tensor(new, key, updated)
    for key in bn_keys:
        value = donor[key]
        assert torch.is_tensor(value) and (not value.is_quantized)
        new[key] = value.clone()
    return (new, audit)


def assert_fixed_grids(state, baseline):
    for k, v in baseline.items():
        if k.endswith('.scale') or k.endswith('.zero_point'):
            assert torch.equal(state[k], v), f'Activation grid changed: {k}'
        if torch.is_tensor(v) and v.is_quantized:
            x = state[k]
            assert x.q_scale() == v.q_scale() and x.q_zero_point() == v.q_zero_point()
    a = state[HEAD][0]
    b = baseline[HEAD][0]
    assert a.q_scale() == b.q_scale() and a.q_zero_point() == b.q_zero_point()


def state_fingerprint(sd):
    h = hashlib.sha256()

    def visit(v):
        if torch.is_tensor(v):
            h.update(str(v.dtype).encode())
            h.update(str(tuple(v.shape)).encode())
            if v.is_quantized:
                h.update(repr((v.q_scale(), v.q_zero_point())).encode())
                h.update(v.int_repr().contiguous().numpy().tobytes())
            else:
                h.update(v.detach().contiguous().numpy().tobytes())
        elif isinstance(v, tuple):
            for x in v:
                visit(x)
        else:
            h.update(str(v).encode())
    for k, v in sd.items():
        h.update(k.encode())
        visit(v)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_mask(path, params):
    grouped = collections.defaultdict(list)
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            k = row['tensor']
            i = int(row['flat_index'])
            if k not in params:
                raise KeyError(f'Mask tensor is not model state: {k}')
            shape = params[k].shape
            if not 0 <= i < params[k].numel():
                raise ValueError(f'Mask index out of range: {k}:{i}')
            if row.get('tensor_index'):
                coord = tuple((int(x) for x in re.findall('\\d+', row['tensor_index'])))
                if len(coord) != len(shape) or np.ravel_multi_index(coord, tuple(shape)) != i:
                    raise ValueError(f'Flat/tensor index disagree: {path}: {row}')
            grouped[k].append(i)
    result = {}
    for k, indices in grouped.items():
        x = torch.tensor(indices, dtype=torch.long)
        if x.unique().numel() != x.numel():
            raise ValueError(f'Duplicate index in {path}: {k}')
        result[k] = x
    return result


def metrics(y, p):
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, den, out=np.zeros(2), where=den != 0)
    return dict(n=len(y), accuracy=float((y == p).mean()), macro_f1=float(f1.mean()), f1_HH=float(f1[0]), f1_Non_HH=float(f1[1]), tn=int(cm[1, 1]), fp=int(cm[1, 0]), fn=int(cm[0, 1]), tp=int(cm[0, 0]), confusion_matrix_json=json.dumps(cm.tolist()))


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('wb' if binary else 'w', **{} if binary else {'newline': ''}) as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        sync_directory(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json(path, obj):
    atomic_write(path, lambda f: json.dump(obj, f, indent=2, allow_nan=False))


def atomic_csv(path, rows):
    if not rows:
        return

    def write(f):
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    atomic_write(path, write)


STRATEGIES = ('stateless',)
EXPECT_FP32 = True
DEFAULT_SEEDS = 10
OUTPUT = "results/stateless"
PATHS = ((2,), (3,), (1, 3), (2, 3), (1, 2, 3))


def find_runs(source, strategy, seed_count):
    runs = []
    for folder in sorted(source.glob('run_*_seed_*')):
        seed = folder.name.split('_')[-1]
        checkpoints = {
            step: folder / f'{strategy}_step_{step:02}_seed_{seed}.pth'
            for step in (1, 2, 3)
        }
        masks = {
            step: path.with_name(path.stem + '_changed_indices.csv')
            for step, path in checkpoints.items()
        }
        if all(path.is_file() for path in (*checkpoints.values(), *masks.values())):
            runs.append((seed, checkpoints, masks))
    if len(runs) < seed_count:
        raise ValueError(f'{source} has {len(runs)} complete runs. Requested {seed_count}.')
    return runs[:seed_count]


def load_update(path, template):
    state = torch.load(path, weights_only=True, map_location='cpu')
    if EXPECT_FP32:
        for key, value in state.items():
            target = template[key]
            dtype = torch.float32 if target.is_quantized else target.dtype
            if value.dtype != dtype or value.shape != target.shape:
                raise ValueError(f'Unexpected tensor in {path}: {key}')
        state[HEAD] = (state.pop('classifier.1.weight'), state.pop('classifier.1.bias'))
    elif not state[HEAD][0].is_quantized:
        raise ValueError(f'Expected a quantized checkpoint: {path}')
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('training/data'))
    parser.add_argument('--output', type=Path, default=Path(OUTPUT))
    parser.add_argument('--seeds', type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    torch.set_num_threads(2)
    baseline_path = args.data_dir / 'initial_baseline_int8quant.pth'
    baseline = torch.load(baseline_path, weights_only=True, map_location='cpu')
    model = make_native_model(baseline)
    template = logical_template(baseline)
    bn_keys = bn_state_keys(model)
    weights = [v for k, v in template.items() if k.endswith('.weight') and v.is_quantized]
    if len(weights) != 53 or any(w.dtype != torch.qint8 or
                                 w.qscheme() != torch.per_tensor_affine or
                                 w.q_zero_point() != 0 for w in weights):
        raise ValueError('Expected baseline signed INT8 weights with per-tensor scales and zero point 0')
    if template['classifier.1.bias'].dtype != torch.float32:
        raise ValueError('Expected FP32 classifier bias')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    dataset = datasets.ImageFolder(args.data_dir / 'test', transform=transform)
    if dataset.class_to_idx != {'HH': 0, 'Non-HH': 1}:
        raise ValueError(f'Unexpected test classes: {dataset.class_to_idx}')
    batches = [images.to(memory_format=torch.channels_last) for images, _ in
               torch.utils.data.DataLoader(dataset, batch_size=32)]
    labels = np.array(dataset.targets)
    results = []
    for strategy in STRATEGIES:
        runs = find_runs(args.data_dir / strategy / 'frac_0p05', strategy, args.seeds)
        for seed, checkpoints, masks in runs:
            donors = {step: load_update(path, template) for step, path in checkpoints.items()}
            indices = {step: read_mask(path, template) for step, path in masks.items()}
            predictions_by_state = {}
            for steps in PATHS:
                state = baseline
                clipping = []
                for step in steps:
                    state, audit = apply_int8_update(state, donors[step], indices[step], bn_keys)
                    assert_fixed_grids(state, baseline)
                    clipping.append({'step': step, 'tensors': audit})
                signature = state_fingerprint(state)
                if signature not in predictions_by_state:
                    model.load_state_dict(state, strict=True)
                    with torch.inference_mode():
                        logits = torch.cat([model(batch) for batch in batches]).numpy()
                    if not np.isfinite(logits).all():
                        raise ValueError(f'Non-finite logits for {strategy}, seed {seed}, path {steps}')
                    predictions_by_state[signature] = logits
                logits = predictions_by_state[signature]
                row = {'strategy': strategy, 'seed': seed, 'path': ' → '.join(map(str, (0, *steps))),
                       **metrics(labels, logits.argmax(1))}
                directory = args.output / strategy / seed / '-'.join(map(str, steps))
                atomic_write(directory / 'predictions.npz',
                             lambda f: np.savez_compressed(f, logits=logits, targets=labels), binary=True)
                atomic_json(directory / 'result.json', {'metrics': row, 'updates': clipping})
                results.append(row)
                atomic_csv(args.output / 'metrics.csv', results)
                print(f"{strategy} {seed} {row['path']}: {row['accuracy']:.2%}", flush=True)
    summary = []
    for strategy in STRATEGIES:
        for steps in PATHS:
            path = ' → '.join(map(str, (0, *steps)))
            group = [r for r in results if r['strategy'] == strategy and r['path'] == path]
            summary.append({'strategy': strategy, 'path': path, 'runs': len(group), **{
                f'{metric}_{stat}': fn([r[metric] for r in group])
                for metric in ('accuracy', 'macro_f1')
                for stat, fn in [('mean', mean), ('sd', lambda x: stdev(x) if len(x) > 1 else '')]
            }})
    atomic_csv(args.output / 'summary.csv', summary)


if __name__ == '__main__':
    main()
