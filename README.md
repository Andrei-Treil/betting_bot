# Welcome
This project is an attempt to analyze NBA games using neural networks in order to make accurate betting predictions. Currently, this project is able to scrape data for a given NBA season to train a scikit MLP. 

Tools for scraping the necessary data as well as generating features and samples are present in [`util/client.py`](notebooks/util/client.py). For examples on how to use these tools, see the notebook [`mlp_betting.ipynb`](notebooks/mlp_betting.ipynb).

Currently, new work is being done in [`bnn_betting.ipynb`](notebooks/bnn_betting.ipynb), which explores the application of Bayesian Neural Networks for this task

# Structure

## 1. Datasets
All collected datasets are contained within the NBA folder:

### [NBA/games](/NBA/games): 
Game information for individual seasons, consisting of involved teams and points scored in
CSV format:

    [Date, Away Team, Away Points, Home Team, Home Points]

No header line included

---

### [NBA/total](/NBA/total): 
Old data, uses season totals for generating features. Contains 2 sub folders:

#### on_off_stats
  
Pickle files containing on/off statistics for individual players and team totals for a given season
  
#### samps_feats
  
  CSV containing feature vectors for each game (16 stats per vector, representing away_stats - home_stats for a given game): 
                    
    [MOV, SOS, SRS, ORtg, DRtg, Pace, FTr, 3PAr, eFG%, TOV%, ORB%, FT/FGA, eFG%, TOV%, DRB%, FT/FGA]

  For information on how this data is collected, see [client.py](notebooks/util/client.py) and [mlp_betting.ipynb](notebooks/mlp_betting.ipynb)

---

### [NBA/consecutive](/NBA/consec): 
New data, uses cumulative stats up to a given date for generating features. For details on how this data was constructed, see [selen_scraper.py](notebooks/util/selen_scraper.py), [client.py](notebooks/util/client.py), and [scrape_new.ipynb](notebooks/scrape_new.ipynb)

#### conc_feats_samps

Features and samples for NBA games using statistics gathered for each team up to a given day. Feature vectors containe 14 stats, representing away_stats - home_stats:

    [OFFRTG, DEFRTG, NETRTG, AST%, AST/TO, AST_RATIO, OREB%, DREB%, REB%, TOV%, EFG%, TS%, PACE, PIE]

Currently, constructed features are based on data from conc_on_off

#### conc_on_off

Pickle file representing dictionary of the following structure:

```
stats format: [OFFRTG, DEFRTG, NETRTG, AST%, AST/TO, AST_RATIO, OREB%, DREB%, REB%, TOV%, EFG%, TS%, PACE, PIE]

ABR -> dict(date -> list of len 3):
	0: <list> overall (team) stats up to this date
	
	1: <dict> PLAYER NAME -> on court stats

	2: <dict> PLAYER NAME -> off court stats
```

#### conc_stats

Pickle file representing dictionary of the following structure:

```
stats format: [PTS, W, L, WIN%, FGM, FGA, FG%, 3PM, 3PA, 3P%, FTM, FTA, FT%, OREB, DREB, REB, AST, TOV, STL, BLK, PF, +/-]

ABR -> dict(date -> list of len 2):
	0: <list> overall (team) stats up to this date
	
	1: <list> overall (team) stats over ~ last 10 games
```

#### conc_stats_adv (deprecated?)

Pickle file similar to that seen in conc_on_off, however does not contain player on/off data

---

### [NBA/with_bets](/NBA/with_bets): 
Game information for individual seaons, including betting lines

These datasets were generated using [util/client.py](util/client.py) by scraping relevant information from basketball reference and vegas insider

---

## 2. Utils
- [client.py](notebooks/util/client.py)
  - Class for scraping data on an individual season
  - Generates on/off statistics and accounts for injuries
- [selen_scraper.py](notebooks/util/selen_scraper.py)
  - Selenium scraper for getting conseucutive games using process pools
- [NeuralNet.py](notebooks/util/NeuralNet.py)
  - Custom neural network design

## 3. Notebooks
- Currently, 3 notebooks are included in the notebooks folder
  - [mlp_betting.ipynb](notebooks/mlp_betting.ipynb)
    - Scraping NBA data for use on scikit-learn MLPs
    - Makes bets based on MLP predictions
  - [bnn_betting.ipynb](notebooks/bnn_betting.ipynb) <- (CURRENT)
    - Creating Bayesian Neural Networks using [Pyro](https://github.com/pyro-ppl/pyro)
  - [scrape_new.ipynb](notebooks/scrape_new.ipynb)
    - Notebook constructing conseuctive data

# Installing
It is recommended to use a virtual environment running Python >= 3.8 when cloning this repository (for instruction on how to do so see [here](https://docs.python.org/3/library/venv.html)). To install the required dependancies, simply run :
`python -m pip install -r requirements.txt` in the root directory of the project. 
