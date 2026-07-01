<div align='center'>
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="resources/logo-darkmode.png">
        <source media="(prefers-color-scheme: light)" srcset="resources/logo-lightmode.png">
        <img alt="mtax logo" src="resources/logo-lightmode.png" width="50%" height="50%">
    </picture>
</div>
<br/>

[![Pytest Tests](https://github.com/filipnaudot/mtax/actions/workflows/tests.yml/badge.svg)](https://github.com/filipnaudot/mtax/actions/workflows/tests.yml)

Multi-topic argumentative exchange



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
- [ ] Resolution-based stopping
- [ ] Generic/abstract turn-taking class TurnTaking
- [ ] Implement simple fixed order turn-taking function that implements TurnTaking
- [ ] Disclosure-effect measures


## TODO for MTAX Usage

- [ ] Optional agent strategy utilities
