"""Training-only practice plans; optionally use complete exchange sessions."""
import random
import numpy as np


def practice_plan(market, config):
    a,b=market.bounds('2020-01-01',config.train_end)
    a+=78  # Completed history for the initial features.
    if not config.session_aligned:
        if b-a<=max(config.train_steps,config.eval_steps):
            raise ValueError('Training history is shorter than the requested episode length.')
        return dict(training_limits=(a,b-config.train_steps),training_steps=config.train_steps,
                    evaluation=[(int(w),int(w)+config.eval_steps) for w in
                                np.linspace(a,b-config.eval_steps,config.eval_windows,dtype=int)])
    days=np.asarray(market.times.tz_convert('America/New_York').date)
    starts=np.r_[0,np.flatnonzero(days[1:]!=days[:-1])+1]
    ends=np.r_[starts[1:]-1,len(days)-1]
    keep=(starts>=a)&(ends<=b)
    starts,ends=starts[keep],ends[keep]
    learning_days=max(1,round(config.train_steps/78))
    evaluation_days=max(1,round(config.eval_steps/78))
    if len(starts)<max(learning_days,evaluation_days):
        raise ValueError('Not enough complete training sessions for the requested history coverage.')
    return dict(training=[(int(starts[i]),int(ends[i+learning_days-1]))
                          for i in range(len(starts)-learning_days+1)],
                evaluation=[(int(starts[i]),int(ends[i+evaluation_days-1])) for i in
                            np.linspace(0,len(starts)-evaluation_days,config.eval_windows,dtype=int)])


def learning_window(plan):
    if 'training' in plan:
        return random.choice(plan['training'])
    start=random.randint(*plan['training_limits'])
    return start,start+plan['training_steps']
