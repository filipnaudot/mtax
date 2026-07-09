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
