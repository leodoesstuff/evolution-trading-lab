"""One process per experiment, protected against duplicate workers."""
import argparse
import fcntl
import os
import time
from pathlib import Path
from .engine import run_experiment
from .store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--home', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--minutes', type=float)
    args = parser.parse_args()
    folder = Path(args.home) / args.run
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'worker.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        (folder / 'worker.pid').write_text(str(os.getpid()))
        try:
            budget = args.minutes if args.minutes is not None else Store(args.home).load(args.run)['config']['minutes']
            began = time.monotonic()
            state = run_experiment(args.home, args.run, budget)
            # Promotion automatically starts the live loop; it does not need a second click.
            remaining = budget - (time.monotonic() - began) / 60
            if state['status'] == 'paper_ready' and remaining > 0 and Store(args.home).control(args.run) == 'run':
                run_experiment(args.home, args.run, remaining)
        finally:
            (folder / 'worker.pid').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
