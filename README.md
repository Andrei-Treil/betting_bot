# Welcome
This project is an attempt to analyze NBA games using neural networks in order to make accurate betting predictions. Currently, this project is able to scrape data for a given NBA season to train a scikit MLP. 

Tools for scraping the necessary data as well as generating features and samples are present in [`util/client.py`](util/client.py). For examples on how to use these tools, see the notebook [`mlp_betting.ipynb`](notebooks/mlp_betting.ipynb).

Currently, new work is being done in [`bnn_betting.ipynb`](notebooks/bnn_betting.ipynb), which explores the application of Bayesian Neural Networks for this task

# Structure
### Datasets
- All collected datasets are contained within the NBA folder, containing 4 sub folders:
  - [`old_games_inj`](/NBA/old_games_inj): Game information for individual seasons, consisting of involved teams and points scored
  - [`old_on_off_stats`](/NBA/old_on_off_stats): On/off statistics and injury information for each season
  - [`old_samps_feats`](/NBA/old_samps_feats): Features and samples for seasons generated from on/off statistics and adjusted for injury
  - [`with_bets`](/NBA/with_bets): Game information for individual seaons, including betting lines

These datasets were generated using [`util/client.py`](util/client.py) by scraping relevant information from basketball reference and vegas insider

### Utils
- [`client.py`](util/client.py)
  - Class for scraping data on an individual season
  - Generates on/off statistics and accounts for injuries
- [`NeuralNet.py`](util/NeuralNet.py)
  - Custom neural network design

### Notebooks
- Currently, 2 notebooks are included in the notebooks folder
  - [`mlp_betting.ipynb`](notebooks/mlp_betting.ipynb)
    - Scraping NBA data for use on scikit-learn MLPs
    - Makes bets based on MLP predictions
  - [`bnn_betting.ipynb`](notebooks/bnn_betting.ipynb)
    - Creating Bayesian Neural Networks using [Pyro](https://github.com/pyro-ppl/pyro)

# Installing
It is recommended to use a virtual environment running Python >= 3.8 when cloning this repository (for instruction on how to do so see [here](https://docs.python.org/3/library/venv.html). To install the required dependancies, simply run `python -m pip install -r requirements.txt` in the root directory of the project. Note: add the name of your venv folder to the gitignore file to avoid tracking dependencies built. To update the requirements.txt with any new packages run `pip freeze > requirements.txt` in the root directory.
