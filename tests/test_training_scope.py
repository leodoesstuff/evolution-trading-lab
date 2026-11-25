import copy
import hashlib
from pathlib import Path

import pytest
import torch

from evo.config import Config
from evo.data import demo_dataset
from evo.engine import new_run
from evo.store import Store
from evo.training_scope import coverage_config, continue_with_coverage


def test_coverage_is_explicit_and_preserves_holdout_boundaries():
    original = Config(rules_version=2, learn_every=32)
    broad = coverage_config(original, 'Broad')
    deep = coverage_config(original, 'Deep')
    assert (broad.train_steps, broad.eval_steps, broad.eval_windows) == (4680, 1560, 6)
    assert (deep.train_steps, deep.eval_steps, deep.eval_windows) == (9360, 3120, 8)
    assert broad.train_end == original.train_end == '2024-01-01'
    assert broad.validation_end == original.validation_end == '2025-01-01'
    assert broad.test_end == original.test_end == '2026-01-01'
    assert broad.learn_every == original.learn_every
    assert original.train_steps == 780  # Legacy checkpoints keep their original defaults.


def test_continuation_keeps_learned_bots_and_source_but_resets_comparisons(tmp_path):
    dataset = demo_dataset(tmp_path/'data', Config().symbols)
    home = tmp_path/'runs'
    cfg = Config(rules_version=2, population=5, generations=5,
                 train_steps=8, eval_steps=8, eval_windows=1)
    source = new_run(home, dataset, cfg)
    store = Store(home)
    saved = store.load(source)
    saved.update(generation=2, cursor=2, status='paused',
                 results=[{'score': 99}], hall=[{'old': 'comparison'}])
    saved['bots'][0]['version'] = 17
    saved['bots'][0]['weights']['actor.bias'].fill_(0.123)
    # Exercise inheritance of actual learning state, not just initialized weights.
    saved['bots'][0]['optimizer']['state'] = {0: {'step': torch.tensor(3.)}}
    store.save(source, saved)
    before = hashlib.sha256((home/source/'checkpoint.pt').read_bytes()).hexdigest()
    continued = continue_with_coverage(home, source, 'Quick')
    result = store.load(continued)
    assert continued != source
    assert result['generation'] == 2 and result['cursor'] == 0
    assert result['results'] == result['hall'] == result['validation'] == []
    assert result['test'] is result['champion'] is result['paper'] is None
    assert result['status'] == 'ready' and result['phase'] == 'training'
    assert result['config']['train_steps'] == 780
    assert result['continuation']['source_run'] == source
    assert torch.equal(result['rng']['torch'], saved['rng']['torch'])
    assert result['rng']['python'] == saved['rng']['python']
    for old, new in zip(saved['bots'], result['bots']):
        assert (new['id'], new['version']) == (old['id'], old['version'])
        assert all(torch.equal(old['weights'][k], new['weights'][k]) for k in old['weights'])
    assert result['bots'][0]['optimizer']['state'][0]['step'].item() == 3
    assert hashlib.sha256((home/source/'checkpoint.pt').read_bytes()).hexdigest() == before
    assert store.rows('SELECT * FROM metrics WHERE run=?', (continued,)).empty
    with pytest.raises(ValueError, match='longer'):
        continue_with_coverage(home, continued, 'Quick')
    slow=continue_with_coverage(home,continued,'Quick',decision_bars=12)
    slow_state=store.load(slow)
    assert slow_state['config']['decision_bars']==12
    assert slow_state['config']['train_steps']==result['config']['train_steps']
    assert slow_state['config']['death_drawdown']==result['config']['death_drawdown']
    assert slow_state['config']['fee_bps']==result['config']['fee_bps']
    assert slow_state['config']['min_roundtrips']==result['config']['min_roundtrips']
    ended = copy.deepcopy(saved)
    ended['phase'] = 'finished'
    store.save(source, ended)
    with pytest.raises(ValueError, match='completed results'):
        continue_with_coverage(home, source)


def test_sidebar_offers_broad_history_by_default_for_real_data(tmp_path, monkeypatch):
    import json
    import evo.config as settings
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(settings, 'DATA', tmp_path)
    path = demo_dataset(tmp_path/'data'/'iex', Config().symbols)
    meta = json.loads((path/'metadata.json').read_text())
    meta['synthetic'] = False
    (path/'metadata.json').write_text(json.dumps(meta))
    app = AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run(timeout=30)
    assert not app.exception
    control = next(s for s in app.selectbox if s.label == 'History per generation')
    assert control.value == 'Broad'
    assert next(s for s in app.selectbox if s.label=='Strategy decision interval · minutes').value==60
    assert any('60 trading days' in c.value and '6 periods' in c.value for c in app.caption)
