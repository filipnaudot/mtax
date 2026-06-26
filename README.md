<div align='center'>
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="resources/logo-darkmode.png">
        <source media="(prefers-color-scheme: light)" srcset="resources/logo-lightmode.png">
        <img alt="mtax logo" src="resources/logo-lightmode.png" width="50%" height="50%">
    </picture>
</div>
<br/>

# mtax
Multi-topic argumentative exchange

## Intended Usage

```python
from mtax import Agent, ExchangeConfig, MTAX


machine = Agent(name="machine", model=...)
human = Agent(name="human", model=...)

exchange = MTAX(
    agents=[machine, human],
    topics=["recommend_x", "recommend_y", "recommend_z"],
    config=ExchangeConfig(max_rounds=10),
)
exchange.run()
result = exchange.result()
```
