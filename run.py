"""CLI for reproducible experiments and exporting their audit trail."""
import argparse
import json
from pathlib import Path
from evo.config import Config, DATA
from evo.data import Alpaca, demo_dataset
from evo.engine import new_run
from evo.store import Store
from evo.training_scope import COVERAGE, coverage_config
from evo.champion_seed import seed_from_champion


def main():
    parser = argparse.ArgumentParser(description='Evolution Trading Lab — local simulations only')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('demo', help='Generate synthetic demo data (never promotes)')
    sub.add_parser('download', help='Download real Alpaca IEX history using environment credentials')
    new = sub.add_parser('new', help='Create an experiment checkpoint')
    new.add_argument('--dataset', default='demo')
    new.add_argument('--population', type=int, default=100)
    new.add_argument('--generations', type=int, default=100)
    new.add_argument('--seed', type=int, default=7)
    new.add_argument('--minutes', type=int, default=15)
    new.add_argument('--coverage', choices=list(COVERAGE), help='Default: Broad for real history, Quick for the synthetic demo')
    new.add_argument('--decision-minutes',type=int,choices=[5,15,30,60],default=60)
    new.add_argument('--champion-run',help='Seed a separate run from this experiment’s saved champion')
    run = sub.add_parser('run', help='Run/resume one bounded session')
    run.add_argument('id')
    run.add_argument('--minutes', type=float, default=15)
    control = sub.add_parser('control')
    control.add_argument('id')
    control.add_argument('action', choices=['pause', 'stop', 'emergency'])
    export = sub.add_parser('export')
    export.add_argument('id')
    args = parser.parse_args()
    cfg = Config()
    if args.command == 'demo':
        print(demo_dataset(DATA / 'data' / 'demo', cfg.symbols))
    elif args.command == 'download':
        print(Alpaca().download(DATA / 'data' / 'alpaca-iex', cfg.symbols, print))
    elif args.command == 'new':
        path = Path(args.dataset)
        path = path if path.is_dir() else DATA / 'data' / args.dataset
        cfg = Config(rules_version=2,population=args.population, generations=args.generations,
                     seed=args.seed, minutes=args.minutes,decision_bars=args.decision_minutes//5,stop_on_extinction=True,
                     training_selection='active_survivors',session_aligned=True)
        synthetic=json.loads((path/'metadata.json').read_text()).get('synthetic',False)
        cfg=coverage_config(cfg,args.coverage or ('Quick' if synthetic else 'Broad'))
        print(seed_from_champion(DATA/'runs',args.champion_run,cfg,path.resolve()) if args.champion_run else new_run(DATA / 'runs', path.resolve(), cfg))
    elif args.command == 'run':
        import subprocess
        import sys
        Store(DATA / 'runs').request(args.id, 'run')
        subprocess.run([sys.executable, '-m', 'evo.worker', '--home', str(DATA / 'runs'),
                        '--run', args.id, '--minutes', str(args.minutes)], check=True)
    elif args.command == 'control':
        Store(DATA / 'runs').request(args.id, args.action)
    elif args.command == 'export':
        store = Store(DATA / 'runs')
        folder = DATA / 'runs' / args.id
        for table in ('trades', 'equity', 'metrics', 'events'):
            store.rows(f'SELECT * FROM {table} WHERE run=?', (args.id,)).to_csv(folder / f'{table}.csv', index=False)
        print(folder)


if __name__ == '__main__':
    main()
