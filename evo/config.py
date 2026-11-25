from dataclasses import dataclass, asdict
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('EVO_HOME', ROOT))


@dataclass
class Config:
    symbols: tuple = ('AAPL', 'MSFT', 'AMZN', 'JPM', 'XOM')
    rules_version: int = 1  # Missing in old checkpoints: preserve the original overnight rules.
    min_roundtrips: int = 3
    min_mean_exposure: float = 0.05
    min_active_day_fraction: float = 0.4
    max_daily_loss: float = 0.03
    warmup_generations: int = 3
    seed: int = 7
    population: int = 100
    generations: int = 100
    initial_cash: float = 10000.0
    max_weight: float = 0.20
    death_drawdown: float = 0.15
    fee_bps: float = 1.0
    slippage_bps: float = 5.0
    participation: float = 0.01
    mutation_std: float = 0.025
    learning_rate: float = 0.0003
    train_steps: int = 780
    eval_steps: int = 390
    eval_windows: int = 3
    learn_every: int = 1
    decision_bars: int = 1  # Missing in old checkpoints: preserve five-minute decisions.
    stop_on_extinction: bool = False
    training_selection: str = 'profit_after_warmup'  # Preserve saved experiments.
    session_aligned: bool = False
    minutes: int = 15
    train_end: str = '2024-01-01'
    validation_end: str = '2025-01-01'
    test_end: str = '2026-01-01'

    def validate(self):
        if self.training_selection not in ('profit_after_warmup','active_survivors'):
            raise ValueError('Unsupported training selection policy.')
        if self.training_selection=='active_survivors' and self.rules_version<2:
            raise ValueError('Active-survivor training requires day-trading activity and risk checks.')
        if self.decision_bars not in (1, 3, 6, 12):
            raise ValueError('Decision interval must be 5, 15, 30 or 60 minutes.')
        if self.rules_version not in (1,2):raise ValueError('Unsupported strategy rules version.')
        if self.min_roundtrips<1 or not 0<self.min_mean_exposure<=1 or not 0<self.min_active_day_fraction<=1:
            raise ValueError('Activity thresholds must be positive and fractions at most one.')
        if not 0<self.max_daily_loss<=self.death_drawdown or self.warmup_generations<0:
            raise ValueError('Invalid daily loss or learning warmup.')
        if self.learn_every not in (1, 8, 32):
            raise ValueError('Learning batch must be 1, 8 or 32 bars.')
        if not 5 <= self.population <= 500:
            raise ValueError('Population must be between 5 and 500.')
        if not 1 <= self.generations <= 1000:
            raise ValueError('Generations must be between 1 and 1000.')
        if len(self.symbols) == 0 or len(set(self.symbols)) != len(self.symbols):
            raise ValueError('Supply unique symbols.')
        if not 0 < self.max_weight <= 0.2 or not 0 < self.death_drawdown <= 0.15:
            raise ValueError('Risk limits must be positive and at most the planned caps.')
        if min(self.initial_cash, self.learning_rate, self.train_steps, self.eval_steps,
               self.eval_windows, self.minutes, self.participation) <= 0:
            raise ValueError('Cash, learning, run length, and liquidity settings must be positive.')
        if min(self.fee_bps, self.slippage_bps, self.mutation_std) < 0:
            raise ValueError('Costs and mutation cannot be negative.')
        if not self.train_end < self.validation_end < self.test_end:
            raise ValueError('Date boundaries must be chronological.')

    def to_dict(self):
        return asdict(self)
