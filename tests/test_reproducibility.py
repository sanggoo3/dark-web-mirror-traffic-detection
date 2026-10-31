import unittest

import numpy as np

from darkmirror_repro.engine_core import choose_threshold
from darkmirror_repro.features import light_features
from darkmirror_repro.io import ROOT, KINDS, csv, dataset, parts, sequence
from darkmirror_repro.preprocessing import encode
from darkmirror_repro.reproduce import audit_hpo, cutoff, latency_summary


class ReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = dataset()

    def test_metadata_and_padding(self):
        d = self.data
        sessions = csv(ROOT / 'data/metadata/sessions.csv')
        sites = csv(ROOT / 'data/metadata/sites_and_domain_groups.csv')
        self.assertEqual(len(sites), 46)
        self.assertEqual(sites.domain_id.nunique(), 44)
        self.assertTrue((sites.sessions == 100).all())
        np.testing.assert_array_equal(sessions.domain_id, d['groups'])
        np.testing.assert_array_equal(sessions.label, d['y'])
        self.assertEqual(int(d['retained_count'].min()), 532)
        self.assertFalse((sequence(d, 'included_526') == 0).any())

    def test_exact_handshake_slice(self):
        np.testing.assert_array_equal(sequence(self.data, 'post506_control')[:, 0], self.data['prefix'][:, 20:526])
        only = sequence(self.data, 'only20_padded526')
        self.assertTrue((only[:, :, 20:] == 0).all())
        np.testing.assert_array_equal(only[:, 0, :20], self.data['prefix'][:, :20])

    def test_splits(self):
        for kind in KINDS:
            tested = []
            for fold in range(1, 6):
                p = parts(ROOT, kind, fold)
                tested.extend(p['test'])
                np.testing.assert_array_equal(np.sort(np.concatenate(list(p.values()))), np.arange(4600))
                for inner in range(1, 4):
                    q = parts(ROOT, kind, fold, inner)
                    self.assertEqual(set(np.concatenate(list(q.values()))), set(p['fit']))
                    self.assertFalse(set(q['fit']) & set(q['validation']))
            np.testing.assert_array_equal(np.sort(tested), np.arange(4600))

    def test_threshold_tie_rule(self):
        rng = np.random.default_rng(17)
        for probabilities in [np.zeros(20), np.ones(20), rng.random(20)]:
            y = np.tile([0, 1], 10)
            original, _ = choose_threshold(y, probabilities)
            self.assertEqual(cutoff(y, probabilities), original)

    def test_features(self):
        d = self.data
        with np.load(ROOT / 'data/processed/light_statistics_526.npz') as data:
            for i in [0, 100, 2300, 4599]:
                np.testing.assert_array_equal(light_features(d['prefix'][i, :526], d['relative_times_s'][i, :526]), data['light'][i])

    def test_encoding(self):
        np.testing.assert_array_equal(encode([1, -1], 4), [1, -1, 0, 0])
        np.testing.assert_array_equal(encode([1, -1, 1], 2), [1, -1])
        with self.assertRaises(ValueError):
            encode([0, 1])

    def test_selection_and_latency(self):
        self.assertEqual(len(audit_hpo(ROOT)), 100)
        timing = latency_summary(csv(ROOT / 'results/recorded/latency_calls.csv.gz'))
        self.assertEqual(len(timing), 24)


if __name__ == '__main__':
    unittest.main()
