"""Small actor-critic policies: lifetime learning plus evolutionary weight mutation."""
import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .sim import Book, observation, allocation, step, results
from .cadence import decision_due


class Policy(nn.Module):
    def __init__(self, inputs, stocks):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(inputs, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.actor = nn.Linear(32, stocks * 4)
        self.critic = nn.Linear(32, 1)
        self.stocks = stocks

    def forward(self, x):
        h = self.body(x)
        return self.actor(h).reshape(-1, self.stocks, 4), self.critic(h).squeeze(-1)


@dataclass
class Bot:
    id: str
    parent: str
    birth: int
    policy: Policy
    optimizer: object
    version: int = 0
    alive: bool = True

    @classmethod
    def fresh(cls, id, inputs, config, birth=0, parent=''):
        policy = Policy(inputs, len(config.symbols))
        return cls(id, parent, birth, policy,
                   torch.optim.Adam(policy.parameters(), lr=config.learning_rate))

    def pack(self):
        return dict(id=self.id, parent=self.parent, birth=self.birth,
                    weights=copy.deepcopy(self.policy.state_dict()),
                    optimizer=copy.deepcopy(self.optimizer.state_dict()),
                    version=self.version, alive=self.alive)

    @classmethod
    def unpack(cls, state, inputs, config):
        bot = cls.fresh(state['id'], inputs, config, state['birth'], state['parent'])
        bot.policy.load_state_dict(state['weights'])
        bot.optimizer.load_state_dict(state['optimizer'])
        bot.version, bot.alive = state['version'], state['alive']
        return bot

    def act(self, obs, learning):
        with torch.no_grad():
            logits, _ = self.policy(torch.as_tensor(obs).unsqueeze(0))
            dist = Categorical(logits=logits)
            action = dist.sample() if learning else logits.argmax(-1)
        return action.squeeze(0).numpy()

    def learn(self, obs, action, reward, next_obs, done, steps=1):
        return self.learn_batch([(obs, action, reward, next_obs, done, steps)])

    def learn_batch(self, transitions):
        obs, actions, rewards, next_obs, dones = zip(*(t[:5] for t in transitions))
        durations = torch.as_tensor([t[5] if len(t)>5 else 1 for t in transitions], dtype=torch.float32)
        logits, value = self.policy(torch.as_tensor(np.array(obs)))
        with torch.no_grad():
            _, next_value = self.policy(torch.as_tensor(np.array(next_obs)))
            target = torch.as_tensor(np.array(rewards), dtype=torch.float32) + .99**durations * (1-torch.as_tensor(np.array(dones),dtype=torch.float32)) * next_value
        advantage = target - value
        dist = Categorical(logits=logits)
        logprob = dist.log_prob(torch.as_tensor(np.array(actions))).sum(-1)
        loss = -(logprob * advantage.detach()).mean() + .5 * advantage.square().mean() - .001 * dist.entropy().sum(-1).mean()
        if not torch.isfinite(loss):
            raise ValueError('Non-finite learning loss; bot update blocked.')
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.version += 1

    def child(self, id, generation, config):
        child = Bot.fresh(id, self.policy.body[0].in_features, config, generation, self.id)
        child.policy.load_state_dict(self.policy.state_dict())
        with torch.no_grad():
            for p in child.policy.parameters():
                mask = torch.rand_like(p) < .10
                p.add_(torch.randn_like(p) * mask * config.mutation_std)
        return child


def episode(bot, market, start, end, config, learning=False, interrupted=None):
    """Interruptible at a bot boundary; aborted work is rolled back by the worker."""
    book = Book.fresh(config)
    book.version = bot.version
    book.observe(market.marks[start], market.times[start], config)
    transitions = []
    last_decision = None
    segment = None
    batch_bars = 0
    for i in range(start, end):
        if interrupted and i % 64 == 0 and interrupted():
            return None, book
        if not book.alive:
            break
        obs = observation(market, i, book)
        book.version = bot.version
        deciding = decision_due(last_decision, market.times[i], config)
        if deciding:
            action = bot.act(obs, learning)
            last_decision = market.times[i]
            segment = [obs, action, 0., 0]
        else:
            action = np.full(len(config.symbols), 3)  # Hold actual shares between decisions.
        targets = allocation(action, book, market.marks[i], config)
        before = book.equity(market.marks[i])
        old_dd = book.max_drawdown
        after = step(market, i + 1, book, targets, config, force_close=i+1==end, hold=not deciding)
        if learning:
            reward = 100 * np.log(max(after, 1e-8) / max(before, 1e-8)) - 10 * max(0, book.max_drawdown - old_dd)
            done = not book.alive or i + 1 == end
            segment[2] += .99**segment[3] * float(reward)
            segment[3] += 1
            batch_bars += 1
            if done or decision_due(last_decision, market.times[i+1], config):
                transitions.append((segment[0], segment[1], segment[2],
                                    observation(market, i+1, book), done, segment[3]))
                if batch_bars >= config.learn_every or done:
                    bot.learn_batch(transitions)
                    transitions.clear()
                    batch_bars = 0
    return results(book, config), book
