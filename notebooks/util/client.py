# import statements
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import time
import csv
from collections import defaultdict
import pickle
import random
import pandas as pd
import itertools
from tqdm.auto import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
import concurrent
import concurrent.futures
import pyro
from pyro.infer import Predictive
import torch
from sklearn.metrics import classification_report,confusion_matrix,make_scorer


class Nba_Season():
    # CONSTANTS
    TEAM_NAME_TO_ABR = {
        "ATLANTA HAWKS": 'ATL',
        "BOSTON CELTICS": 'BOS',
        "BROOKLYN NETS": 'BRK',
        "CHARLOTTE HORNETS": 'CHO',
        "CHICAGO BULLS": 'CHI',
        "CLEVELAND CAVALIERS": 'CLE',
        "DALLAS MAVERICKS": 'DAL',
        "DENVER NUGGETS": 'DEN',
        "DETROIT PISTONS": 'DET',
        "GOLDEN STATE WARRIORS": 'GSW',
        "HOUSTON ROCKETS": 'HOU',
        "INDIANA PACERS": 'IND',
        "LOS ANGELES CLIPPERS": 'LAC',
        "LOS ANGELES LAKERS": 'LAL',
        "MEMPHIS GRIZZLIES": 'MEM',
        "MIAMI HEAT": 'MIA',
        "MILWAUKEE BUCKS": 'MIL',
        "MINNESOTA TIMBERWOLVES": 'MIN',
        "NEW ORLEANS PELICANS": 'NOP',
        "NEW YORK KNICKS": 'NYK',
        "OKLAHOMA CITY THUNDER": 'OKC',
        "ORLANDO MAGIC": 'ORL',
        "PHILADELPHIA 76ERS": 'PHI',
        "PHOENIX SUNS": 'PHO',
        "PORTLAND TRAIL BLAZERS": 'POR',
        "SACRAMENTO KINGS": 'SAC',
        "SAN ANTONIO SPURS": 'SAS',
        "TORONTO RAPTORS": 'TOR',
        "UTAH JAZZ": 'UTA',
        "WASHINGTON WIZARDS": 'WAS',

        # DEPRECATED TEAMS
        # "CHARLOTTE BOBCATS": 'CHA',
        # "KANSAS CITY KINGS": 'KCK',
        # "NEW JERSEY NETS": 'NJN',
        # "NEW ORLEANS HORNETS": 'NOH',
        # "NEW ORLEANS/OKLAHOMA CITY HORNETS": 'NOK',
        # "SEATTLE SUPERSONICS": 'SEA',
        # "ST. LOUIS HAWKS": 'STL',
        # "VANCOUVER GRIZZLIES": 'VAN',
        # "WASHINGTON BULLETS": 'WSB',
    }

    MONTH_TO_NUM = {
        "Jan" : "01",
        "Feb" : "02",
        "Mar" : "03",
        "Apr" : "04",
        "May" : "05",
        "Jun" : "06",
        "Jul" : "07",
        "Aug" : "08",
        "Sep" : "09",
        "Oct" : "10",
        "Nov" : "11",
        "Dec" : "12",
    }

    def __init__(self, start_year, end_year, team_stats=None, team_on_off=None, features=None, samples=None, conc=None):
        '''
        `start_year`: String representing start year of NBA season
        `end_year`: String representing end year of NBA season
        `team_stats`: Dict mapping team abr to stats
        `team_on_off`: Dict mapping team abr to dict of player names mapping to on-off stats
        `features`: List of unnormalized features for games representing the season
        `samples`: List of samples corresponding to features
        `conc`: Dict from NBA/conc_on_off, holds stats for team and on/off court stats for players:

            stats = GP MIN OFFRTG DEFRTG NETRTG	AST% AST/TO	AST_RATIO OREB%	DREB% REB%	TOV% EFG% TS% PACE PIE

            conc[ABR] = dict[dates] -> [ [TEAM Stats], dict[player] -> stats (on court), dict[player] -> stats (on court)]
        '''
        self.start_year = start_year
        self.end_year = end_year
        self.team_stats = defaultdict(list) if team_stats is None else team_stats
        self.team_on_off = defaultdict(dict) if team_on_off is None else team_on_off
        self.features = features
        self.samples = samples
        self.conc = conc
    
    # get team stats for a given season from https://www.basketball-reference.com/teams/
    def get_team_stats(self,team_abr):
        '''
        `team_abr`: string abbreviation from basketball-reference
        `end_year`: string representing end year of season to get stats
        '''
        URL = "https://www.basketball-reference.com/teams/{team_abr}/{end_year}.html".format(team_abr=team_abr,end_year=self.end_year)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.find('div',id='all_team_misc')
        dec = results.decode_contents()
        new_soup = BeautifulSoup(dec,'lxml')
        comment = new_soup.find(text=lambda text:isinstance(text, Comment))
        com_soup = BeautifulSoup(comment,'lxml')
        table = com_soup.find_all('td')

        team_res = []

        # MOV	SOS	SRS	ORtg	DRtg	Pace	FTr	3PAr	eFG%	TOV%	ORB%	FT/FGA	eFG%	TOV%	DRB%	FT/FGA	
        for i in range(4,20):
            team_res.append(float(table[i].text))

        return team_res
    

    # populate team stats based on season year
    def pop_team_stats(self):
        for abr in self.TEAM_NAME_TO_ABR.values():
            self.team_stats[abr] = self.get_team_stats(abr)


    # get team on off stats
    def get_on_off(self,team_abr):
        url = ("https://www.basketball-reference.com/teams/{team_abr}/{end_year}/on-off/").format(team_abr=team_abr,end_year=self.end_year)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.find('table',id='on_off')
        on_off_list = list(results.find_all('th',scope='row'))
        on_off_dict = defaultdict(list)

        for i in range(0,len(on_off_list)-2,3):
            name = on_off_list[i].text.upper()
            on_off_dict[name] = list(on_off_list[i+2].previous.stripped_strings)[1:]
        
        return on_off_dict


    # populate team on off stats
    def pop_team_on_off(self):
        for abr in self.TEAM_NAME_TO_ABR.values():
            self.team_on_off[abr] = self.get_on_off(abr)


    # calculate injuray impact based on on-off values
    def calc_injury_impact(self,injured,home_abr,away_abr,date):
        '''
        Calculate the injury impact for a game based on on/off stats
        `injured`: dict mapping ABR -> list of injured players
        `home_abr`: string abbreviation of home team
        `away_abr`: string abbreviation of away team
        `date`: date as it appears in CSV, EX: Tue Oct 24 2023
        '''

        home_injured = injured[home_abr]
        away_injured = injured[away_abr]

        if self.conc is None:
            home_stats = self.team_stats[home_abr]
            away_stats = self.team_stats[away_abr]
            # map available on-off stats to respective index in team stats, ignore rest for now
            # mapping [on_off_idx,team_stats_idx]
            affected_stats_idx = [[1,8],[2,10],[3,14],[8,9],[9,5],[10,3],[18,13]]
            # eFG%	ORB%	DRB%	TRB%	AST%	STL%	BLK%	TOV%	Pace	ORtg	eFG%	ORB%	DRB%	TRB%	AST%	STL%	BLK%	TOV%	Pace	ORtg
            # 8      10       14    NAN     NAN      NAN     NAN     9       5       3       NAN     NAN     NAN     NAN     NAN     NAN     NAN     13      NAN     NAN
            for player in home_injured:
                on_off =  self.team_on_off[home_abr][player]
                if len(on_off) == 0:
                    continue
                
                weight = float(on_off[0].strip('%'))/100
                
                for pair in affected_stats_idx:
                    home_stats[pair[1]] -= (weight * float(on_off[pair[0]]))

            for player in away_injured:
                on_off =  self.team_on_off[away_abr][player]
                if len(on_off) == 0:
                    continue
                
                weight = float(on_off[0].strip('%'))/100
                
                for pair in affected_stats_idx:
                    away_stats[pair[1]] -= (weight * float(on_off[pair[0]]))
        else:
            # GP MIN OFFRTG DEFRTG NETRTG	AST% AST/TO	AST_RATIO OREB%	DREB% REB%	TOV% EFG% TS% PACE PIE
            home_team_conc = self.conc[home_abr][date]
            away_team_conc = self.conc[away_abr][date]
            home_stats = home_team_conc[0][2:]
            away_stats = away_team_conc[0][2:]

            for player in home_injured:
                player_on = home_team_conc[1][player]
                player_off = home_team_conc[2][player]

                if len(player_on) == 0 or len(player_off) == 0:
                    continue

                min_ratio = float(player_on[1] / (home_team_conc[0][1]))

                for i in range(len(home_stats)):
                    on_off = player_on[i] - player_off[i]
                    home_stats[i] -= (min_ratio * on_off)

            for player in away_injured:
                player_on = away_team_conc[1][player]
                player_off = away_team_conc[2][player]

                if len(player_on) == 0 or len(player_off) == 0:
                    continue
                
                min_ratio = float(player_on[1] / (away_team_conc[0][1]))

                for i in range(2,len(away_stats)):
                    on_off = player_on[i] - player_off[i]
                    away_stats[i] -= (min_ratio * on_off)

        return np.subtract(away_stats,home_stats)


    def check_injured(self,box_score_page,home_abr,away_abr,date):
        '''
        Checks list of injured players for a given game and features
        `box_score_page`: string representing URL for a given game
        `home_team`: string abbreviation of home team
        `away_team`: string abbreviation of away team
        `date`: date as it appears in CSV, EX: Tue Oct 24 2023
        '''
        page = requests.get(box_score_page)
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.find_all('strong')
        player_links = results[3].previous.find_all('a')

        curr_team = " "
        players_dict = {home_abr : [], away_abr : []}

        for link in player_links:
            player_name = link.text.upper()
            prev = list(link.previous_sibling.stripped_strings)

            if prev[0] == home_abr:
                curr_team = home_abr

            elif prev[0] == away_abr:
                curr_team = away_abr

            players_dict[curr_team].append(player_name)

        return self.calc_injury_impact(players_dict,home_abr,away_abr,date)
    
    def check_injured_selen(self,links):
            '''
            Implementation of check_injured using Selenium
            `box_score_page`: string representing URL for a given game
            `home_team`: string abbreviation of home team
            `away_team`: string abbreviation of away team
            `date`: date as it appears in CSV, EX: Tue Oct 24 2023
            '''
            chrome_ops = webdriver.ChromeOptions()
            chrome_ops.add_argument("--headless=new")
            chrome_ops.page_load_strategy = 'none'
            driver = webdriver.Chrome(options=chrome_ops)
            out = []
            try:
                for i in tqdm(range(9, len(links), 10)):
                    k = 0
                    batch = []
                    for j in range(i-9,i+1):
                        link = links[j]
                        batch.append(link)
                        box_score_page = link[0]
                        driver.get(box_score_page)
                        k += 1
                        if k < 10: # change range step to control how many tabs are opened in a batch, current set to 10
                            driver.execute_script("window.open()")
                            driver.switch_to.window(driver.window_handles[k])

                    window_list = driver.window_handles
                    k = 0
                    time.sleep(1)

                    for window in window_list:
                        driver.switch_to.window(window)
                        html = driver.page_source
                        soup = BeautifulSoup(html, 'html.parser')

                        results = soup.find_all('strong')
                        player_links = results[3].previous.find_all('a')

                        home_abr = batch[k][1]
                        away_abr = batch[k][2]
                        date = batch[k][3]

                        k += 1

                        curr_team = " "
                        players_dict = {home_abr : [], away_abr : []}

                        for link in player_links:
                            player_name = link.text.upper()
                            prev = list(link.previous_sibling.stripped_strings)

                            if prev[0] == home_abr:
                                curr_team = home_abr

                            elif prev[0] == away_abr:
                                curr_team = away_abr

                            players_dict[curr_team].append(player_name)
                    
                        out.append(self.calc_injury_impact(players_dict,home_abr,away_abr,date))
                        if len(driver.window_handles) > 1:
                            driver.close()

                batch = []
                k = 0
                remaining = len(links) % 10
                if remaining > 0:
                    for i in range(len(links) - remaining, len(links)):
                        link = links[i]
                        batch.append(link)
                        box_score_page = link[0]
                        driver.get(box_score_page)
                        k += 1
                        if k < remaining: # change range step to control how many tabs are opened in a batch, current set to 10
                            driver.execute_script("window.open()")
                            driver.switch_to.window(driver.window_handles[k])

                    for window in window_list:
                        driver.switch_to.window(window)
                        html = driver.page_source
                        soup = BeautifulSoup(html, 'html.parser')

                        results = soup.find_all('strong')
                        player_links = results[3].previous.find_all('a')

                        home_abr = batch[k][1]
                        away_abr = batch[k][2]
                        date = batch[k][3]

                        k += 1

                        curr_team = " "
                        players_dict = {home_abr : [], away_abr : []}

                        for link in player_links:
                            player_name = link.text.upper()
                            prev = list(link.previous_sibling.stripped_strings)

                            if prev[0] == home_abr:
                                curr_team = home_abr

                            elif prev[0] == away_abr:
                                curr_team = away_abr

                            players_dict[curr_team].append(player_name)
                    
                        out.append(self.calc_injury_impact(players_dict,home_abr,away_abr,date))
                        if len(driver.window_handles) > 1:
                            driver.close()

            except Exception as e:
                driver.quit()
                print(e)
            driver.quit()
            return out


    def generate_features(self,file_path,set_categorical=True,use_selenium=False):
        '''
        Returns lists containing features, samples for a given season
        `file_path`: path of CSV containing games for a season
        `set_categorical`: boolean flag to determine wether to create categorical samples or continuous (default True)
            EX: Los Angeles Lakers,107,Denver Nuggets,119
            True: sample = [0,1]
            False: sample = [107,119]
        `use_selenium`: boolean flag to use selenium for scraping (suggested if requests being blocked)
        '''
        
        features = []
        samples = []
        links = []
        line_count = 1
        fail_count = 0

        # construct data set, consisting of team misc stats as features and win/loss as samples
        with open(file_path, mode='r') as f:
            lines = list(csv.reader(f))
            for date,away_team,away_pt,home_team,home_pt in tqdm(lines):
                try:
                    date_list = date.split()
                    month = self.MONTH_TO_NUM[date_list[1]]
                    day = date_list[2] if len(date_list[2]) == 2 else "0{day}".format(day=date_list[2])
                    year = date_list[3]
                    
                    if set_categorical:
                        results = [1,0] if away_pt > home_pt else [0,1]
                    else:
                        results = [int(away_pt),int(home_pt)]
                    
                    samples.append(results)

                    # time.sleep(random.uniform(1,3))
                    # get box score page
                    box_score_page = "https://www.basketball-reference.com/boxscores/{YEAR}{MO}{DA}0{HOME}.html".format(YEAR=year,MO=month,DA=day,HOME=self.TEAM_NAME_TO_ABR[home_team.upper()])
                    if use_selenium:
                        links.append([box_score_page,self.TEAM_NAME_TO_ABR[home_team.upper()],self.TEAM_NAME_TO_ABR[away_team.upper()],date])
                    else:
                        feats = self.check_injured(box_score_page,self.TEAM_NAME_TO_ABR[home_team.upper()],self.TEAM_NAME_TO_ABR[away_team.upper()],date)
                        time.sleep(random.randint(1,3))
                        features.append(feats)
                        
                    line_count += 1

                except Exception as e:
                    print(e)
                    print("Failed at count: {count} for file: {file_name} LINK: {link}".format(count=line_count,file_name=file_path,link=box_score_page))
                    fail_count += 1
                    line_count += 1
                    if fail_count > 5:
                        self.features = features
                        self.samples = samples
                        print("Too many failures, terminating feature generation")
                        return features,samples
                    continue

            if use_selenium:
                all_batches = []
                num_workers = 5
                batch_size = len(links) // num_workers

                for i in range(0,len(links),int(batch_size)):
                    all_batches.append(links[i:i+batch_size])

                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    features = list(executor.map(self.check_injured_thread,all_batches))
                
        self.features = features
        self.samples = samples
        return features,samples
    

    def add_bet_info(self,games_path,out_path):
        '''
        Populates CSV with betting info from https://www.vegasinsider.com/nba/odds/las-vegas/
        `file_path`: path of CSV containing games
        `out_path`: output path for games with betting info
        '''

        fail_count = 0
        # store odds in dict to avoid accessing same page, map date -> team -> odds
        odds_dict = defaultdict(dict)
        new_games = []
        header = ["date","away_team","away_pt","home_team","home_pt","home_spread","home_total","home_ml","away_spread","away_total","away_ml"]

        with open(games_path, mode='r') as f:
            lines = csv.reader(f)
            for date,away_team,away_pt,home_team,home_pt in lines:
                try:
                    date_list = date.split()
                    month = self.MONTH_TO_NUM[date_list[1]]
                    day = date_list[2] if len(date_list[2]) == 2 else "0{day}".format(day=date_list[2])
                    year = date_list[3]
                    # TODO: change to 1d here?
                    results = [1,0] if away_pt > home_pt else [0,1]

                    game_with_odds = [date,away_team,away_pt,home_team,home_pt]
                    
                    # populate odds for that day and store in odds_dict
                    if len(odds_dict[date]) == 0:
                        team_dict = defaultdict(list)
                        odds_page = "https://www.vegasinsider.com/nba/odds/las-vegas/?date={YEAR}-{MO}-{DA}&table=moneyline".format(YEAR=year,MO=month,DA=day)
                        page = requests.get(odds_page)
                        soup = BeautifulSoup(page.content, 'html.parser')
                        result = soup.find('table')
                        odds_table = list(result.stripped_strings)

                        # 4 = Home vs, 5 = Away, 6-11 alternating home and away for each team
                        # Headers: Home, Away, Spread, Total, Moneyline
                        for i in range(4,len(odds_table)-7,8):
                            home_t = odds_table[i][0:-3].upper()
                            home_st = [odds_table[j] for j in range(i+2,i+8,2)]
                            away_t = odds_table[i+1].upper()
                            away_st = [odds_table[j] for j in range(i+3,i+8,2)]
                            team_dict[home_t] = home_st
                            team_dict[away_t] = away_st
                        
                        odds_dict[date] = team_dict
                    
                    home_fixed = home_team.upper().split()[-1]
                    away_fixed = away_team.upper().split()[-1]

                    if home_fixed == 'BLAZERS':
                        home_fixed = 'TRAIL BLAZERS'
                    if away_fixed == 'BLAZERS':
                        away_fixed = 'TRAIL BLAZERS'

                    game_with_odds.extend(odds_dict[date][home_fixed])
                    game_with_odds.extend(odds_dict[date][away_fixed])
                    new_games.append(game_with_odds)
                    

                except Exception as e:
                    print(e)
                    #print(box_score_page)
                    fail_count += 1
                    time.sleep(random.randint(1, 5))
                    if fail_count > 5:
                        print("Too many failures, terminating...")
                        games_df = pd.DataFrame(new_games)
                        games_df.to_csv(out_path,header=header,index=False)
                        return new_games
                    continue
        
        games_df = pd.DataFrame(new_games)
        games_df.to_csv(out_path,header=header,index=False) 
        return new_games


    # populate team stats and on off stats for new season
    def pop_const_new(self,save_folder='../NBA/on_off_stats/'):
        '''
        Populate team stats and on-off data, exports to .pkl with name '{start}-{end}_team_stats.pkl'
        `save_folder`: leading file path to store outputs, default: '../NBA/on_off_stats/'
        '''
        self.pop_team_stats()

        with open('{save}{start}-{end}_team_stats.pkl'.format(save=save_folder,start=self.start_year,end=self.end_year), 'wb') as f:
            pickle.dump(self.team_stats, f)

        time.sleep(30)

        self.pop_team_on_off()

        with open('{save}{start}-{end}_on_off.pkl'.format(save=save_folder,start=self.start_year,end=self.end_year), 'wb') as f:
            pickle.dump(self.team_on_off, f)
    
    def save_data(self,save_path=''):
        '''
        Save features and samples to SAVE_PATH
        '''
        np.savetxt('{save_path}{start_year}-{end_year}_nba_features.csv'.format(save_path=save_path,start_year=self.start_year,end_year=self.end_year), self.features, delimiter=',')
        np.savetxt('{save_path}{start_year}-{end_year}_nba_samples.csv'.format(save_path=save_path,start_year=self.start_year,end_year=self.end_year), self.samples, delimiter=',')

def kelly(home_pred,away_pred,home_line,away_line,max_bet=100,diff_thresh=0.05,diff_cap=0.25):
    '''
    Applies kelly critereon based on features and moneyline data
    home_pred: Prediction from MLP for home team
    away_pred: Prediction from MLP for away team
    home_line: Moneyline for home team
    away_line: Moneyline for away team
    diff_thresh: Minimum difference between prediction and implied odds
    diff_cap: Maximum difference between prediciton and implied odds
    '''
    bet_amount = 0
    to_win = 0

    log_home = (home_pred - (home_pred * away_pred)) / (home_pred + away_pred - (2*home_pred*away_pred))
    log_away = (away_pred - (home_pred * away_pred)) / (home_pred + away_pred - (2*home_pred*away_pred))

    # calculate ratio and implied for home
    home_line_adj = home_line
    away_line_adj = away_line
    if home_line < 0:
        home_line_adj *= -1
        home_line_adj /= 100
        home_ratio = 1/(home_line_adj)
        implied_home = home_line_adj/(1+home_line_adj)
    else:
        home_line_adj /= 100
        home_ratio = home_line_adj
        implied_home = 1/(home_line+1)

    # calculate ratio and implied for away
    if away_line < 0:
        away_line_adj *= -1
        away_line_adj /= 100
        away_ratio = 1/(away_line_adj)
        implied_away = away_line_adj/(1+away_line_adj)
    else:
        away_line_adj /= 100
        away_ratio = away_line_adj
        implied_away = 1/(away_line_adj+1)
    
    diff_home = log_home - implied_home
    diff_away = log_away - implied_away

    kelly_home = log_home - (log_away/home_ratio)
    kelly_away = log_away - (log_home/away_ratio)

    prob = 0

    # make bets, negative if away team bet
    if diff_home > diff_away and diff_home > diff_thresh and diff_home < diff_cap:
        bet_amount = (max_bet*kelly_home)
        if home_line < 0:
            to_win = bet_amount/((home_line*-1)/100)
        else:
            to_win = bet_amount/((home_line)/100)
        prob = home_pred

    
    elif diff_away > diff_home and diff_away > diff_thresh and diff_away < diff_cap:
        bet_amount = (max_bet*kelly_away)
        if away_line < 0:
            to_win = -1*bet_amount/((away_line*-1)/100)
        else:
            to_win = -1*bet_amount/((away_line)/100)
        prob = away_pred

    return bet_amount,to_win,prob

def BNN_kelly(preds,actual,money_lines,one_hot=False,diff_thresh=0.05,diff_cap=0.25):
    money_made = 0
    money_risked = 0
    correct = 0
    guessed = 0
    team_bet = []
    amount = []
    gained = []
    probs = []     

    for i in range(len(preds)):
        if one_hot:
            away_pred = preds[i][0]
            home_pred = preds[i][1]
        else:
            home_pred = preds[i]
            away_pred = 1 - home_pred
        home_ml = money_lines[i][7]
        away_ml = money_lines[i][10]

        to_bet,to_win,prob = kelly(home_pred,away_pred,home_ml,away_ml,diff_thresh=diff_thresh,diff_cap=diff_cap)
        probs.append(prob)
        money_risked += to_bet

        curr_gained = 0

        if to_win < 0:
            team_bet.append('Away')
            amount.append(to_bet)
            guessed += 1
            curr_gained = -1*to_bet
            if actual[i] == 1:
                correct += 1
                curr_gained = (-1*to_win)
                #money_made += curr_gained
        elif to_win > 0:
            team_bet.append('Home')
            amount.append(to_bet)
            guessed += 1
            curr_gained = -1*to_bet
            if actual[i] == 0:
                correct += 1
                curr_gained = to_win
                #money_made += curr_gained
        else:
            team_bet.append(0)
            amount.append(0)

        gained.append(curr_gained)

        if curr_gained > 0:
            money_made += curr_gained

    return correct,guessed,team_bet,probs,amount,gained

def pred_performance(train_preds: Predictive,test_preds: Predictive,y_train: torch.Tensor,
                     y_test: torch.Tensor,use_obs=True,use_ret=True,categorical=True):
    '''
    Prints confusion matrix and classification reports of a Predictive pyro object on training and test data, using "obs" and "_RETURN" flags

    `train_preds`: Tensor containing Pyro prediction object on training data
    `test_preds`: Tensor containing Pyro prediction object on test data
    `y_train`: Tensor containing outputs for x_train
    `y_test`: Tensor containingoutputs for x_test
    `use_obs`: Boolean flag to use the "obs" flag for predictive
    `use_ret`: Boolean flag to use the "_RETURN" flag for predictive
    `categorical`: Boolean flag to determine whether samples are categorical
    '''

    y_train_1d = [0 if j[0] == 0 else 1 for j in y_train] # [0,1] -> home win -> 0 indicates home win, 1 indicates away
    y_test_1d = [0 if j[0] == 0 else 1 for j in y_test]

    if use_obs == False and use_ret == False:
        print('ERROR: set "use_obs" or "use_ret" to True')
        return

    print('---TRAINING SET---')

    if use_obs:
        if categorical:
            obs_preds = train_preds['obs'].float().mean(axis=1).float().mean(axis=0)
        else:
            obs_preds = train_preds['obs'].float().mean(axis=0)

        adj_train_preds = [0 if p[0] < p[1] else 1 for p in obs_preds]
        print('OBS:')
        print('TN, FP, FN, TP')
        print(confusion_matrix(y_train_1d,adj_train_preds).ravel())
        print(classification_report(y_train_1d,adj_train_preds))

    if use_ret:
        ret_preds = train_preds['_RETURN'].float().mean(axis=0)
        adj_train_preds = [0 if p[0] < p[1] else 1 for p in ret_preds]
        print('RET:')
        print('TN, FP, FN, TP')
        print(confusion_matrix(y_train_1d,adj_train_preds).ravel())
        print(classification_report(y_train_1d,adj_train_preds))

    print('---TEST SET---')

    if use_obs:
        if categorical:
            obs_preds = test_preds['obs'].float().mean(axis=1).float().mean(axis=0)
        else:
            obs_preds = test_preds['obs'].float().mean(axis=0)

        adj_test_preds = [0 if p[0] < p[1] else 1 for p in obs_preds]
        print('OBS:')
        print('TN, FP, FN, TP')
        print(confusion_matrix(y_test_1d,adj_test_preds).ravel())
        print(classification_report(y_test_1d,adj_test_preds))
    
    if use_ret:
        ret_preds = test_preds['_RETURN'].float().mean(axis=0)
        adj_test_preds = [0 if p[0] < p[1] else 1 for p in ret_preds]
        print('RET:')
        print('TN, FP, FN, TP')
        print(confusion_matrix(y_test_1d,adj_test_preds).ravel())
        print(classification_report(y_test_1d,adj_test_preds))

def make_bets(train_preds,test_preds,bet_data_train,bet_data_test,bet_samps_train,
              bet_samps_test,use_obs=True,use_ret=True,diff_thresh=0.05,diff_cap=0.25,categorical=True):
    
    '''
    Places bets using predictions made by a Pyro predictive object using the kelly critereon

    `train_preds`: Tensor containing Pyro prediction object on training data
    `test_preds`: Tensor containing Pyro prediction object on test data
    `bet_data_train`: Arraylike containing betting data for training
    `bet_data_test`: Arraylike containing betting data for testing
    `bet_samps_train`: Arraylike containing training samples
    `bet_samps_test`: Arraylike containing test samples
    `use_obs`: Boolean flag to use the "obs" flag for predictive
    `use_ret`: Boolean flag to use the "_RETURN" flag for predictive
    `diff_thresh`: Minimum difference between implied odds and predicted
    `diff_thresh`: Maximum difference between implied odds and predicted
    `categorical`: Boolean flag to determine whether samples are categorical
    '''
    
    if use_obs == False and use_ret == False:
        print('ERROR: set "use_obs" or "use_ret" to True')
        return
    
    bet_samps_train_1d = [0 if j[0] < j[1] else 1 for j in bet_samps_train]
    bet_samps_test_1d = [0 if j[0] < j[1] else 1 for j in bet_samps_test]


    print('PREDICTIONS ON 2022-2023 DATA (SEEN IN TRAINING)')
    if use_obs:
        if categorical:
            new_y_pred = train_preds['obs'].float().mean(axis=1).float().mean(axis=0)
        else:
            new_y_pred = train_preds['obs'].float().mean(axis=0)

        print('Using OBS:')
        print(f'max confidence: {new_y_pred.max():.2f}')
        correct,guessed,team_bet,probs,amount,gained = BNN_kelly(new_y_pred,bet_samps_train_1d,bet_data_train[1:],one_hot=True,diff_thresh=diff_thresh,diff_cap=diff_cap)
        print(f'correct: {correct}')
        print(f'guessed: {guessed}')
        print(f'risked: {sum(amount)}')
        print(f'made: {sum(gained)}')
        print(f'ROI: {(sum(gained)/sum(amount)):.2f}\n')
    
    if use_ret:
        new_y_pred = train_preds['_RETURN'].float().mean(axis=0)
        print('Using RET:')
        print(f'max confidence: {new_y_pred.max():.2f}')
        correct,guessed,team_bet,probs,amount,gained = BNN_kelly(new_y_pred,bet_samps_train_1d,bet_data_train[1:],one_hot=True,diff_thresh=diff_thresh,diff_cap=diff_cap)
        print(f'correct: {correct}')
        print(f'guessed: {guessed}')
        print(f'risked: {sum(amount)}')
        print(f'made: {sum(gained)}')
        print(f'ROI: {(sum(gained)/sum(amount)):.2f}\n')

    print('PREDICTIONS ON 2023-2024 DATA (UNSEEN)')
    if use_obs:
        print('Using OBS:')
        if categorical:
            new_y_pred = test_preds['obs'].float().mean(axis=1).float().mean(axis=0)
        else:
            new_y_pred = test_preds['obs'].float().mean(axis=0)
            
        print(f'max confidence: {new_y_pred.max():.2f}')
        correct,guessed,team_bet,probs,amount,gained = BNN_kelly(new_y_pred,bet_samps_test_1d,bet_data_test[1:],one_hot=True,diff_thresh=diff_thresh,diff_cap=diff_cap)
        print(f'correct: {correct}')
        print(f'guessed: {guessed}')
        print(f'risked: {sum(amount)}')
        print(f'made: {sum(gained)}')
        print(f'ROI: {(sum(gained)/sum(amount)):.2f}\n')

    if use_ret:
        print('Using RET:')
        new_y_pred = test_preds['_RETURN'].float().mean(axis=0)
        print(f'max confidence: {new_y_pred.max():.2f}')
        correct,guessed,team_bet,probs,amount,gained = BNN_kelly(new_y_pred,bet_samps_test_1d,bet_data_test[1:],one_hot=True,diff_thresh=diff_thresh,diff_cap=diff_cap)
        print(f'correct: {correct}')
        print(f'guessed: {guessed}')
        print(f'risked: {sum(amount)}')
        print(f'made: {sum(gained)}')
        print(f'ROI: {(sum(gained)/sum(amount)):.2f}\n')
        print()