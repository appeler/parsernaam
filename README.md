# Parsernaam: ML-Assisted Name Parser

[![image](https://github.com/appeler/parsernaam/workflows/test/badge.svg)](https://github.com/appeler/parsernaam/actions?query=workflow%3Atest)
[![image](https://img.shields.io/pypi/v/parsernaam.svg)](https://pypi.python.org/pypi/parsernaam)
[![image](https://static.pepy.tech/badge/parsernaam)](https://pepy.tech/project/parsernaam)

Most common name parsers use crude pattern matching and the sequence of
strings, e.g., the last word is the last name, to parse names. This
approach is limited and fragile, especially for Indian names. We take a
machine-learning approach to the problem. Using the large voter
registration data in India and the US, we build machine-learning-based name
parsers that predict whether the string is a first or last name.

For Indian electoral rolls, we assume the last name is the word in the
name that is shared by multiple family members. (We table the expansion
to include compound last names\-\--extremely rare in India\-\--till the
next iteration.)

# Gradio App.

[parsernaam on HF](https://huggingface.co/spaces/sixtyfold/parsernaam)

# Installation

``` bash
pip install parsernaam
```

# General API

The general API is as follows:

    # Import the library
    from parsernaam.parsernaam import ParseNames

    positional arguments:
      df                 dataframe with Names to parse (with column name 'name')

    # example
    df = pd.DataFrame({'name': ['Jan', 'Nicholas Turner', 'Petersen', 'Nichols Richard', 'Piet',
                                         'John Smith', 'Janssen', 'Kim Yeon']})
    df = ParseNames.parse(df)
    print(df.to_markdown())

    |    | name            | parsed_name                                                                   |
    |---:|:----------------|:------------------------------------------------------------------------------|
    |  0 | Jan             | {'name': 'Jan', 'type': 'first', 'prob': 0.6769440174102783}                  |
    |  1 | Nicholas Turner | {'name': 'Nicholas Turner', 'type': 'first_last', 'prob': 0.9990382194519043} |
    |  2 | Petersen        | {'name': 'Petersen', 'type': 'last', 'prob': 0.5342262387275696}              |
    |  3 | Nichols Richard | {'name': 'Nichols Richard', 'type': 'last_first', 'prob': 0.9998832941055298} |
    |  4 | Piet            | {'name': 'Piet', 'type': 'first', 'prob': 0.5381495952606201}                 |
    |  5 | John Smith      | {'name': 'John Smith', 'type': 'first_last', 'prob': 0.9975730776786804}      |
    |  6 | Janssen         | {'name': 'Janssen', 'type': 'first', 'prob': 0.5929554104804993}              |
    |  7 | Kim Yeon        | {'name': 'Kim Yeon', 'type': 'last_first', 'prob': 0.9987115859985352}        |

# Data

The model is trained on names from the Florida Voter Registration Data
from early 2022. The data are available on the [Harvard
Dataverse](http://dx.doi.org/10.7910/DVN/UBIG3F)

# Authors

Rajashekar Chintalapati and Gaurav Sood

# Contributing

Contributions are welcome. Please open an issue if you find a bug or
have a feature request.

## 🔗 Adjacent Repositories

- [appeler/naamkaran](https://github.com/appeler/naamkaran) — generative model for names
- [appeler/ethnicolr2](https://github.com/appeler/ethnicolr2) — Ethnicolr implementation with new models in pytorch
- [appeler/instate](https://github.com/appeler/instate) — instate: predict the state of residence from last name using the indian electoral rolls
- [appeler/pranaam](https://github.com/appeler/pranaam) — pranaam: predict religion based on name
- [appeler/graphic_names](https://github.com/appeler/graphic_names) — Infer the gender of person with a particular first name using Google image search and Clarifai
