# API Reference

This section provides detailed documentation of the parsernaam API.

## Core Classes

```{eval-rst}
.. automodule:: parsernaam.parse
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: parsernaam.naam
   :members:
   :undoc-members:
   :show-inheritance:
```

## Model Architecture

```{eval-rst}
.. automodule:: parsernaam.model
   :members:
   :undoc-members:
   :show-inheritance:
```

## Utilities

```{eval-rst}
.. automodule:: parsernaam.utils
   :members:
   :undoc-members:
   :show-inheritance:
```

## Configuration

```{eval-rst}
.. automodule:: parsernaam.config
   :members:
   :undoc-members:
   :show-inheritance:
```

## Package Information

```{eval-rst}
.. automodule:: parsernaam
   :members:
   :undoc-members:
   :show-inheritance:
```

## Quick Reference

### Main Functions

The primary interface for name parsing:

```{eval-rst}
.. autofunction:: parsernaam.parse.ParseNames.parse
```

### Core Classes

```{eval-rst}
.. autoclass:: parsernaam.parse.ParseNames
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: parsernaam.naam.Parsernaam  
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: parsernaam.model.LSTM
   :members:
   :undoc-members:
```

### Usage Examples

Basic parsing:

```python
from parsernaam.parse import ParseNames
import pandas as pd

df = pd.DataFrame({'name': ['John Smith', 'Jane Doe']})
results = ParseNames.parse(df)
```

Model architecture:

```python
from parsernaam.model import LSTM

# Model automatically loaded and cached
model = LSTM(input_size=100, hidden_size=128, output_size=2, num_layers=1)
```

Command line utilities:

```python
from parsernaam.utils import get_args

args = get_args(['input.csv', '-o', 'output.csv', '-n', 'name'], 
                'Parse names', 'Example usage', 'out.csv')
```