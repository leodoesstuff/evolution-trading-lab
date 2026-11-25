"""Seed a separate research run from a qualified, locally saved champion."""
import copy
import json
from dataclasses import replace
from pathlib import Path
import torch
import numpy as np

from .config import Config
from .engine import new_run, rng_state, write_report
from .model import Bot
from .store import Store
from .status_labels import qualified_champion
from .data import load_dataset


def latest_champion(home):
    for path in sorted(Path(home).glob('*/report.json'),reverse=True):
        try:
            report=json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            continue  # An active worker may be publishing its next report.
        if report.get('champion') and report.get('promotion_reason')=='Passed validation and untouched final test.':
            return dict(run=report['run'],bot=report['champion'])
    return None


def transfer_champion(packed, source_config, config, feature_inputs):
    if tuple(source_config.symbols)!=tuple(config.symbols):
        raise ValueError('Champion transfer requires the same stocks in the same order.')
    old_inputs=feature_inputs+len(config.symbols)+2+(2 if source_config.rules_version>=2 else 0)
    new_inputs=feature_inputs+len(config.symbols)+2+(2 if config.rules_version>=2 else 0)
    if new_inputs<old_inputs:
        raise ValueError('Removing learned model inputs is not supported.')
    adapted=copy.deepcopy(packed)
    if adapted['weights']['body.0.weight'].shape[1]!=old_inputs:
        raise ValueError('Saved champion input shape does not match its configuration.')
    if old_inputs!=new_inputs:
        def insert_clock(tensor):
            zeros=tensor.new_zeros((tensor.shape[0],new_inputs-old_inputs))
            return torch.cat((tensor[:,:feature_inputs],zeros,tensor[:,feature_inputs:]),dim=1)
        adapted['weights']['body.0.weight']=insert_clock(adapted['weights']['body.0.weight'])
        parameter=adapted['optimizer']['param_groups'][0]['params'][0]
        for key,value in adapted['optimizer']['state'].get(parameter,{}).items():
            if isinstance(value,torch.Tensor) and value.ndim==2 and value.shape[1]==old_inputs:
                adapted['optimizer']['state'][parameter][key]=insert_clock(value)
    adapted.update(id='g000-b000',parent=packed['id'],birth=0,alive=True)
    for group in adapted['optimizer']['param_groups']:
        group['lr']=config.learning_rate
    return Bot.unpack(adapted,new_inputs,config)


def seed_from_champion(home, source_run, config=None, dataset=None):
    store=Store(home)
    source=store.load(source_run)
    if not qualified_champion(source):
        raise ValueError('Choose a run with a qualified historical champion.')
    if config is None:
        config=replace(Config(**source['config']),rules_version=2,training_selection='active_survivors',
                       session_aligned=True,decision_bars=12,stop_on_extinction=True,
                       train_steps=4680,eval_steps=1560,eval_windows=6,population=100,generations=100)
    config.validate()
    if tuple(source['config']['symbols'])!=tuple(config.symbols):
        raise ValueError('Champion seeding requires the same stock universe.')
    # A different dataset/scaler would change the meaning of inherited weights.
    if dataset is not None and Path(dataset).resolve()!=Path(source['dataset']).resolve():
        raise ValueError('Choose the champion’s original dataset to preserve its feature scaling.')
    market=load_dataset(source['dataset'],config.symbols,config.train_end)
    if market.meta['fingerprint']!=source['fingerprint'] or any(
            not np.array_equal(market.meta[key],source['meta'][key]) for key in ('scaler_mean','scaler_std')):
        raise ValueError('Champion dataset or feature scaling changed; transfer was blocked.')
    prototype=transfer_champion(source['champion'],Config(**source['config']),config,
                                len(source['meta']['scaler_mean'])+len(config.symbols))
    run=new_run(home,source['dataset'],config)
    state=store.load(run)
    seeded=max(1,int(config.population*.8))
    bots=[prototype.pack()]
    for i in range(1,seeded):
        bots.append(prototype.child(f'g000-b{i:03}',0,config).pack())
    bots.extend(state['bots'][seeded:])
    state.update(bots=bots,rng=rng_state(),seed_source=dict(source_run=source_run,
                 bot=source['champion']['id'],version=source['champion']['version'],
                 exact_copies=1,mutated_descendants=seeded-1,fresh_bots=config.population-seeded,
                 source_rules=source['config'].get('rules_version',1),
                 target_rules=config.rules_version,
                 holdout_note='Seed champion was selected using the old validation/test periods. These are reused research periods; automatic promotion still requires an unused holdout.'))
    store.save(run,state)
    write_report(store,state)
    store.event(run,'seeded_from_champion',**state['seed_source'])
    return run
