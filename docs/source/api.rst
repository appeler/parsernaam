API Reference
=============

This section provides detailed documentation of the parsernaam API.

Core Classes
------------

.. automodule:: parsernaam.parse
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: parsernaam.naam
   :members:
   :undoc-members:
   :show-inheritance:

Model Architecture
------------------

.. automodule:: parsernaam.model
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: parsernaam.utils
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. automodule:: parsernaam.config
   :members:
   :undoc-members:
   :show-inheritance:

Package Information
-------------------

.. automodule:: parsernaam
   :members:
   :undoc-members:
   :show-inheritance:

Quick Reference
---------------

Main Functions
~~~~~~~~~~~~~~

The primary interface for name parsing:

.. autofunction:: parsernaam.parse.ParseNames.parse

Core Classes
~~~~~~~~~~~~

.. autoclass:: parsernaam.parse.ParseNames
   :members:
   :undoc-members:

.. autoclass:: parsernaam.naam.Parsernaam  
   :members:
   :undoc-members:

.. autoclass:: parsernaam.model.LSTM
   :members:
   :undoc-members:

Usage Examples
~~~~~~~~~~~~~~

Basic parsing:

.. code-block:: python

   from parsernaam.parse import ParseNames
   import pandas as pd
   
   df = pd.DataFrame({'name': ['John Smith', 'Jane Doe']})
   results = ParseNames.parse(df)

Model architecture:

.. code-block:: python

   from parsernaam.model import LSTM
   
   # Model automatically loaded and cached
   model = LSTM(input_size=100, hidden_size=128, output_size=2, num_layers=1)

Command line utilities:

.. code-block:: python

   from parsernaam.utils import get_args
   
   args = get_args(['input.csv', '-o', 'output.csv', '-n', 'name'], 
                   'Parse names', 'Example usage', 'out.csv')