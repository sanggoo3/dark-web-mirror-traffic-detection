from contextlib import contextmanager
from pathlib import Path
import hashlib
import time
import warnings

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import models
from .engine_core import NeuralEngineCore, set_seed
from .io import ROOT, dataset, read_json, save_json, sequence, sha


@contextmanager
def rocket_precision():
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
        yield


class Fusion526(torch.nn.Module):
    def __init__(self, base, family):
        super().__init__()
        self.features, self.classifier = base.features, base.classifier
        if family == 'TCN':
            previous = self.classifier[1]
            self.classifier[1] = torch.nn.Linear(previous.in_features + 16, previous.out_features)
        else:
            previous = self.classifier
            self.classifier = torch.nn.Linear(previous.in_features + 16, previous.out_features)

    def forward(self, packed):
        direction = self.features(packed[:, :, :526]).flatten(1)
        return self.classifier(torch.cat([direction, packed[:, 0, 526:]], dim=1)).squeeze(1)


class PaperEngine(NeuralEngineCore):
    def __init__(self, root=ROOT, out=None, heartbeat=None):
        if not torch.cuda.is_available():
            raise RuntimeError('Recorded training requires CUDA; verify/reproduce work without CUDA.')
        self.root = Path(root)
        self.out = Path(out or self.root / 'artifacts')
        self.out.mkdir(parents=True, exist_ok=True)
        self.data = dataset(root)
        self.y, self.groups = self.data['y'], self.data['groups']
        self.models = models
        self.models.SEQ_LEN = 526
        self.original_data_hash = sha(self.root / 'data/processed/direction_sequences.npz')
        self.gpu_inputs = {}
        self.feature_arrays = {}
        self.heartbeat = heartbeat or (lambda **kwargs: None)
        torch.set_num_threads(2)
        threadpool_limits(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        self.configure('included_526')

    def configure(self, condition, fit=None):
        self.mode, self.scaler = condition, None
        if condition == 'light_fusion526':
            if fit is None:
                raise ValueError('Fusion requires training indices, never validation/test fitting.')
            with np.load(self.root / 'data/processed/light_statistics_526.npz') as data:
                light = data['light']
            self.scaler = StandardScaler().fit(light[fit])
            standardized = self.scaler.transform(light).astype(np.float32)
            self.packed = np.concatenate([sequence(self.data, 'included_526'), standardized[:, None, :]], axis=2)
        else:
            self.packed = sequence(self.data, condition)
        self.packed = np.ascontiguousarray(self.packed, dtype=np.float32)
        self.data_hash = hashlib.sha256(self.packed.tobytes() + condition.encode()).hexdigest()
        self.gpu_inputs.clear()

    def input(self, condition):
        assert condition == self.mode
        return self.packed

    def build(self, name, config):
        base = super().build(name, config)
        if self.mode == 'light_fusion526':
            assert name in ['TCN', 'ResNet1D']
            return Fusion526(base, name).cuda()
        return base

    @staticmethod
    def kernels(seed, count):
        rng = np.random.default_rng(seed)
        result = []
        for length in [7, 9, 11, 15]:
            weight = rng.normal(size=(count // 4, 1, length)).astype(np.float32)
            weight -= weight.mean(axis=2, keepdims=True)
            bias = rng.uniform(-1, 1, count // 4).astype(np.float32)
            result.append((weight, bias))
        return result

    @staticmethod
    def rocket_transform(values, kernels, batch_size=128):
        device = [(torch.from_numpy(w).cuda(), torch.from_numpy(b).cuda()) for w, b in kernels]
        output = []
        with rocket_precision(), torch.inference_mode():
            for start in range(0, len(values), batch_size):
                xb = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device='cuda')
                pieces = []
                for weight, bias in device:
                    act = F.conv1d(xb, weight, bias, padding=weight.shape[-1] // 2)
                    pieces.extend([act.amax(dim=2), (act > 0).float().mean(dim=2)])
                output.append(torch.cat(pieces, dim=1).cpu().numpy())
        return np.concatenate(output).astype(np.float32)

    def fit(self, name, config, seed, fit, validation, folder, condition='included_526', test=None):
        if test is not None:
            raise ValueError('Test evaluation must occur after the complete choice/threshold lock.')
        if name != 'ROCKET-style':
            return super().fit(name, config, seed, fit, validation, folder, condition, test=None)
        assert condition == self.mode == 'included_526'
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        identity = dict(model=name, config=config, seed=seed, condition=condition, data_sha256=self.data_hash,
                        fit_sha256=hashlib.sha256(np.asarray(fit, dtype=np.int64).tobytes()).hexdigest(),
                        validation_sha256=hashlib.sha256(np.asarray(validation, dtype=np.int64).tobytes()).hexdigest())
        if (folder / 'result.json').exists():
            result = read_json(folder / 'result.json')
            assert result['identity'] == identity
            return result
        started = time.perf_counter()
        set_seed(seed)
        key = seed, config['n_kernels'], self.data_hash
        if key not in self.feature_arrays:
            self.feature_arrays[key] = self.rocket_transform(self.packed, self.kernels(seed, config['n_kernels']))
        values = self.feature_arrays[key]
        scaler = StandardScaler().fit(values[fit])
        classifier = LogisticRegression(C=config['C'], class_weight=config['class_weight'],
                                        solver='lbfgs', max_iter=3000, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            classifier.fit(scaler.transform(values[fit]), self.y[fit])
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError('LogisticRegression did not converge.')
        predict = lambda indices: classifier.predict_proba(scaler.transform(values[indices]))[:, 1]
        joblib.dump(dict(scaler=scaler, classifier=classifier, kernels=self.kernels(seed, config['n_kernels']),
                         configuration=config, seed=seed, fit_indices=fit, identity=identity,
                         input_length=526), folder / 'model.joblib')
        self.save_scores(folder / 'validation_scores.csv', validation, predict(validation))
        result = dict(identity=identity, fit_samples=len(fit), validation_samples=len(validation),
                      fit_auc=float(roc_auc_score(self.y[fit], predict(fit))),
                      validation_auc=float(roc_auc_score(self.y[validation], predict(validation))),
                      best_epoch=0, epochs_run=0, seconds=time.perf_counter() - started)
        result['train_validation_gap'] = result['fit_auc'] - result['validation_auc']
        save_json(folder / 'result.json', result)
        return result

    def runtime(self, name, config, folder):
        folder = Path(folder)
        if name == 'ROCKET-style':
            bundle = joblib.load(folder / 'model.joblib')
            weights = [(torch.from_numpy(w).cuda(), torch.from_numpy(b).cuda()) for w, b in bundle['kernels']]
            def probability(value):
                with rocket_precision(), torch.inference_mode():
                    xb = torch.as_tensor(value, dtype=torch.float32, device='cuda')
                    pieces = []
                    for weight, bias in weights:
                        act = F.conv1d(xb, weight, bias, padding=weight.shape[-1] // 2)
                        pieces.extend([act.amax(dim=2), (act > 0).float().mean(dim=2)])
                    transformed = torch.cat(pieces, dim=1).cpu().numpy()
                return float(bundle['classifier'].predict_proba(bundle['scaler'].transform(transformed))[0, 1])
            return probability, bundle
        bundle = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
        network = self.build(name, config).eval()
        network.load_state_dict(bundle['state_dict'])
        def probability(value):
            with torch.inference_mode():
                x = torch.as_tensor(value, dtype=torch.float32, device='cuda')
                if name in ['GRU', 'BiGRU', 'LSTM']:
                    x = x.transpose(1, 2)
                return float(torch.sigmoid(network(x)).item())
        return probability, network
