"""Separate process and lock namespace for local live-practice competitions."""
import argparse
import fcntl
import os
from pathlib import Path
from .arena import run_practice


def main():
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--session',required=True);p.add_argument('--minutes',type=int,default=60);a=p.parse_args()
    folder=Path(a.home)/a.session
    with (folder/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        (folder/'worker.pid').write_text(str(os.getpid()))
        try:run_practice(a.home,a.session,a.minutes)
        finally:(folder/'worker.pid').unlink(missing_ok=True)


if __name__=='__main__':main()
