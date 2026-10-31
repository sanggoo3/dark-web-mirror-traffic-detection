from __future__ import annotations
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .io import save_json
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate(y, probability, threshold):
    predicted = probability >= threshold
    return dict(zip(METRICS, [float(accuracy_score(y, predicted)), float(precision_score(y, predicted, zero_division=0)), float(recall_score(y, predicted, zero_division=0)), float(f1_score(y, predicted, zero_division=0)), float(roc_auc_score(y, probability))]))

def choose_threshold(y, probability):
    records = [dict(Threshold=float(t), **calculate(y, probability, float(t))) for t in np.round(np.arange(0.1, 0.901, 0.01), 2)]
    table = pd.DataFrame(records)
    winner = table.sort_values(['F1-score', 'Accuracy', 'Threshold'], ascending=[False, False, True]).iloc[0]
    return (float(winner.Threshold), table)

class NeuralEngineCore:

    def gpu_input(self, condition, rnn=False):
        key = (condition, rnn)
        if key not in self.gpu_inputs:
            values = self.input(condition)
            if rnn:
                values = values.transpose(0, 2, 1).copy()
            self.gpu_inputs[key] = torch.from_numpy(values).cuda()
        return self.gpu_inputs[key]

    def build(self, name, config):
        torch.backends.cudnn.deterministic = name != 'TCN'
        a, dropout = (config['architecture'], config['dropout'])
        m = self.models
        if name == 'TCN':
            model = m.DirectionTCN(**a, dropout=dropout)
        elif name == 'ResNet1D':
            model = m.DirectionResNet1D(**a, dropout=dropout)
        elif name == '1D-CNN':
            model = m.DirectionOnlyCNN(conv_channels=a['channels'], kernel_size=a['kernel_size'], fc_hidden=a['fc_hidden'], dropout=dropout)
        elif name == 'InceptionTimeLite':
            assert a['kernels'] == [9, 19, 39]
            model = m.InceptionTimeLite(channels=a['channels'], depth=a['depth'], dropout=dropout)
        elif name == 'FCN-1D':
            model = m.FCN1D(**a, dropout=dropout)
        elif name in ['GRU', 'LSTM']:
            model = m.DirectionRNN(cell_type=name.lower(), hidden_size=a['hidden_size'], num_layers=a['num_layers'], dropout=dropout)
        elif name == 'BiGRU':
            model = m.DirectionBiGRU(hidden_size=a['hidden_size'], num_layers=a['num_layers'], dropout=dropout)
        elif name == 'Transformer Encoder':
            model = m.TransformerEncoder1D(d_model=a['d_model'], nhead=a['nhead'], num_layers=a['num_layers'], dropout=dropout, dim_feedforward=a['dim_feedforward'])
        else:
            raise ValueError(name)
        return model.cuda()

    def predict_neural(self, model, name, condition, indices):
        model.eval()
        values = self.gpu_input(condition, name in ['GRU', 'BiGRU', 'LSTM'])
        scores = []
        with torch.inference_mode():
            for start in range(0, len(indices), 128):
                selected = torch.as_tensor(indices[start:start + 128], device='cuda')
                scores.append(torch.sigmoid(model(values[selected])).cpu().numpy())
        return np.concatenate(scores).astype(np.float64)

    def save_scores(self, path, indices, probabilities, threshold=None):
        assert np.isfinite(probabilities).all()
        rows = pd.DataFrame(dict(index=indices, group=self.groups[indices], y_true=self.y[indices], probability=probabilities))
        if threshold is not None:
            rows['threshold'] = threshold
            rows['y_pred'] = (probabilities >= threshold).astype(int)
        rows.to_csv(path, index=False)

    def fit(self, name, config, seed, fit, validation, folder, condition='included_526', test=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        identity = dict(model=name, config=config, seed=seed, condition=condition, data_sha256=self.data_hash, fit_sha256=__import__('hashlib').sha256(np.asarray(fit, dtype=np.int64).tobytes()).hexdigest(), validation_sha256=__import__('hashlib').sha256(np.asarray(validation, dtype=np.int64).tobytes()).hexdigest())
        if test is not None:
            identity['test_sha256'] = __import__('hashlib').sha256(np.asarray(test, dtype=np.int64).tobytes()).hexdigest()
        if (folder / 'result.json').exists():
            result = json.loads((folder / 'result.json').read_text())
            assert result['identity'] == identity
            return result
        started = time.perf_counter()
        set_seed(seed)
        history = []
        model = self.build(name, config)
        values = self.gpu_input(condition, name in ['GRU', 'BiGRU', 'LSTM'])
        targets = torch.as_tensor(self.y, dtype=torch.float32, device='cuda')
        fit_tensor = torch.as_tensor(fit, device='cuda')
        shuffle = torch.Generator().manual_seed(seed + 1009)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best_auc, best_epoch, stale, state = (-np.inf, 0, 0, None)
        for epoch in range(1, 51):
            model.train()
            order = torch.randperm(len(fit), generator=shuffle).cuda()
            losses = []
            for start in range(0, len(fit), config['batch_size']):
                selected = fit_tensor[order[start:start + config['batch_size']]]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(values[selected]), targets[selected])
                assert torch.isfinite(loss), (name, seed, epoch)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                assert torch.isfinite(gradient)
                optimizer.step()
                losses.append(float(loss.detach()))
            probabilities = self.predict_neural(model, name, condition, validation)
            auc = float(roc_auc_score(self.y[validation], probabilities))
            if auc > best_auc:
                best_auc, best_epoch, stale = (auc, epoch, 0)
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_auc=auc))
            self.heartbeat(epoch=epoch, maximum_epochs=50, validation_auc=auc)
            if epoch >= 20 and stale >= 10:
                break
        model.load_state_dict(state)
        temporary = folder / 'model.pt.tmp'
        torch.save(dict(model=name, config=config, seed=seed, condition=condition, input_length=values.shape[1 if name in ['GRU', 'BiGRU', 'LSTM'] else 2], best_epoch=best_epoch, state_dict=state, identity=identity), temporary)
        temporary.replace(folder / 'model.pt')
        epochs = len(history)
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        predict = lambda indices: self.predict_neural(model, name, condition, indices)
        validation_probability = predict(validation)
        fit_probability = predict(fit)
        result = dict(identity=identity, fit_samples=len(fit), validation_samples=len(validation), fit_auc=float(roc_auc_score(self.y[fit], fit_probability)), validation_auc=float(roc_auc_score(self.y[validation], validation_probability)), best_epoch=best_epoch, epochs_run=epochs)
        result['train_validation_gap'] = result['fit_auc'] - result['validation_auc']
        self.save_scores(folder / 'validation_scores.csv', validation, validation_probability)
        if test is not None:
            cutoff, table = choose_threshold(self.y[validation], validation_probability)
            table.to_csv(folder / 'threshold_search.csv', index=False)
            test_probability = predict(test)
            self.save_scores(folder / 'test_scores.csv', test, test_probability, cutoff)
            self.save_scores(folder / 'fit_scores.csv', fit, fit_probability)
            result.update(Threshold=cutoff, **calculate(self.y[test], test_probability, cutoff))
        result['seconds'] = time.perf_counter() - started
        save_json(folder / 'result.json', result)
        if name != 'ROCKET-style':
            del model, optimizer, state
            torch.cuda.empty_cache()
        return result
