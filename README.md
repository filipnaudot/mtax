<div align='center'>
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="resources/logo-darkmode.png">
        <source media="(prefers-color-scheme: light)" srcset="resources/logo-lightmode.png">
        <img alt="mtax logo" src="resources/logo-lightmode.png" width="50%" height="50%">
    </picture>
</div>
<br/>

[![Pytest Tests](https://github.com/filipnaudot/mtax/actions/workflows/tests.yml/badge.svg)](https://github.com/filipnaudot/mtax/actions/workflows/tests.yml)

MTAX provides a structured, formal environment in which multiple agents can debate a set of topics/recommendations.
The framework manages the public argumentative exchange, validates disclosed arguments and relations, updates each agent's private quantitative bipolar argumentation framework (QBAF), and determines when the agents have reached a resolution.

The user is responsible for designing the agents.
Each agent must decide:

1. How to evaluate the strength of arguments it receives.
2. Which private arguments and relations to disclose publicly during the debate.

MTAX is also designed to support optional QBAF-based disclosure strategies.
These strategies can guide what an agent discloses using its argument evaluations, without depending directly on the agent's underlying reasoning process.



## Usage

```python
from mtax import MTAXAgent, ExchangeConfig, MTAX

class MyAgent(MTAXAgent):
    def contribute(self, violation_feedback=None):
        ...

exchange = MTAX(
    agents=[MyAgent("machine"), MyAgent("human")],
    topics=["recommend_x", "recommend_y", "recommend_z"],
    config=ExchangeConfig(max_rounds=10),
)

for state in exchange:
    ...

result = exchange.result()
```


## TODO for Repo

- [x] CI that runs tests


## TODO for MTAX Framework

- [x] Argument, relation, and disclosure schemas
- [x] Bipolar Multitree structure and guardrails
- [x] Atomic multi-relation publication
- [x] Abstract agent interface
- [x] QBAF-Py integration and semantics
- [x] Top-r (and stance) resolution
- [x] Integrate resolution check in MTAX
- [x] Resolution-based stopping
- [ ] Generic/abstract turn-taking class TurnTaking
- [ ] Implement simple fixed order turn-taking function that implements TurnTaking
- [ ] Disclosure-effect measures


## TODO for MTAX Usage

- [ ] Agent strategy utilities
