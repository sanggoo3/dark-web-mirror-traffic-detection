import argparse
import json
from pathlib import Path

from .io import ROOT, verify


def main():
    parser = argparse.ArgumentParser(description='Manuscript-specific direction-sequence reproducibility')
    parser.add_argument('--root', type=Path, default=ROOT)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    report = sub.add_parser('reproduce')
    report.add_argument('--out', type=Path)
    report.add_argument('--skip-bootstrap', action='store_true')
    for name in ['tune', 'train', 'evaluate', 'summarize', 'benchmark']:
        task = sub.add_parser(name)
        task.add_argument('--out', type=Path, required=True)
        if name in ['tune', 'train']:
            task.add_argument('--split', choices=['group_aware', 'session_stratified'], action='append')
            task.add_argument('--model', action='append')
            task.add_argument('--fold', type=int, choices=range(1, 6), action='append')
        if name == 'train':
            task.add_argument('--seed', type=int, choices=[11, 23, 71, 101, 202], action='append')
            task.add_argument('--recorded-choices', action='store_true')
        if name == 'benchmark':
            task.add_argument('--private-manifest', type=Path, required=True)
            task.add_argument('--tshark', default='tshark')
    args = parser.parse_args()
    if args.command == 'verify':
        result = verify(args.root)
    elif args.command == 'reproduce':
        from .reproduce import reproduce
        result = reproduce(args.root, args.out, not args.skip_bootstrap)
    elif args.command == 'benchmark':
        from .latency import benchmark
        result = benchmark(args.out, args.private_manifest, args.root, args.tshark)
    elif args.command == 'summarize':
        from .reproduce import summarize_new
        result = summarize_new(args.out, args.root)
    else:
        from .experiments import tune, train, evaluate
        if args.command == 'tune':
            result = tune(args.out, args.root, args.split, args.model, args.fold)
        elif args.command == 'train':
            result = train(args.out, args.root, args.recorded_choices, args.split, args.model, args.fold, args.seed)
        else:
            result = evaluate(args.out, args.root)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
