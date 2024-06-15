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
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
import concurrent
import concurrent.futures

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

TEAM_ABR_TO_URL = {
        "ATL": 'https://www.nba.com/stats/team/1610612737',
        "BOS": 'https://www.nba.com/stats/team/1610612738',
        "BRK": 'https://www.nba.com/stats/team/1610612751',
        "CHO": 'https://www.nba.com/stats/team/1610612766',
        "CHI": 'https://www.nba.com/stats/team/1610612741',
        "CLE": 'https://www.nba.com/stats/team/1610612739',
        "DAL": 'https://www.nba.com/stats/team/1610612742',
        "DEN": 'https://www.nba.com/stats/team/1610612743',
        "DET": 'https://www.nba.com/stats/team/1610612765',
        "GSW": 'https://www.nba.com/stats/team/1610612744',
        "HOU": 'https://www.nba.com/stats/team/1610612745',
        "IND": 'https://www.nba.com/stats/team/1610612754',
        "LAC": 'https://www.nba.com/stats/team/1610612746',
        "LAL": 'https://www.nba.com/stats/team/1610612747',
        "MEM": 'https://www.nba.com/stats/team/1610612763',
        "MIA": 'https://www.nba.com/stats/team/1610612748',
        "MIL": 'https://www.nba.com/stats/team/1610612749',
        "MIN": 'https://www.nba.com/stats/team/1610612750',
        "NOP": 'https://www.nba.com/stats/team/1610612740',
        "NYK": 'https://www.nba.com/stats/team/1610612752',
        "OKC": 'https://www.nba.com/stats/team/1610612760',
        "ORL": 'https://www.nba.com/stats/team/1610612753',
        "PHI": 'https://www.nba.com/stats/team/1610612755',
        "PHO": 'https://www.nba.com/stats/team/1610612756',
        "POR": 'https://www.nba.com/stats/team/1610612757',
        "SAC": 'https://www.nba.com/stats/team/1610612758',
        "SAS": 'https://www.nba.com/stats/team/1610612759',
        "TOR": 'https://www.nba.com/stats/team/1610612761',
        "UTA": 'https://www.nba.com/stats/team/1610612762',
        "WAS": 'https://www.nba.com/stats/team/1610612764',
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

 # ABR -> dict(date -> [cum_stats,last10])


def get_stats_thread(info):
    '''
    Go through list of dates and populate ALL_TEAMS_CONC
    dates: list of dates similar to `Tue Oct 29 2013`
    base_url: base url of NBA stats page for this team
    abr: team abbreviation
    RETURNS: dict mapping date -> [cum_stats, last10]
    '''
    dates = info[0]
    base_url = info[1]
    options = Options()
    chrome_ops = webdriver.ChromeOptions()
    chrome_ops.add_argument("--headless=new")
    options.headless = True
    driver = webdriver.Chrome(options=options) # requires appropriate webdriver in PATH variables, or pass path to file as arg `executable_path=`
    # driver.implicitly_wait(2)
    wait = WebDriverWait(driver,10,0.1)
    out_dict = defaultdict(list)
    missing = []
    i = 0
    try:
        for date in dates:
            date_list = date.split()
            team_stats = base_url + f'&DateTo={MONTH_TO_NUM[date_list[1]]}%2F{date_list[2]}%2F{date_list[3]}'
            if i < 82:
                team_stats += '&SeasonType=Regular+Season&Split=lastn'
            else:
                team_stats += '&SeasonType=Playoffs&Split=lastn'
            driver.get(team_stats)
            # load JS
            try:
                wait.until(EC.visibility_of_any_elements_located((By.TAG_NAME, 'td')))
            except Exception as e:
                # print(f'URL: {team_stats} caused exception: {e}')
                missing.append(date)
                continue
            
            html = driver.page_source

            soup = BeautifulSoup(html, 'html.parser')
            results = soup.find_all('table')

            # 24 stats total from i = 1 to 24, only care about 3 - 24
            # GP	MIN	PTS	W	L	WIN%	FGM	FGA	FG%	3PM	3PA	3P%	FTM	FTA	FT%	OREB	DREB	REB	AST	TOV	STL	BLK	PF	+/-
            cumalitive = results[2].find_all('td') # cum stats to this point
            cum_stats = []
            for e in cumalitive[3:]: # only care about last ten games?
                cum_stats.append(float(e.text))

            per_n = results[3].find_all('tr') # every 10 games ranges
            last_10 = []

            # if last 10 range includes at least 5 games, use those, else use previous 10 range
            if float(per_n[-1].find_all('td')[1].text) >= 5 or i < 10 or len(per_n) < 2:
                for e in per_n[-1].find_all('td')[3:]:
                    last_10.append(float(e.text))
                
            else:
                for e in per_n[-2].find_all('td')[3:]:
                    last_10.append(float(e.text))
            if i % 20 == 0:
                print(len(out_dict))
            out_dict[date].append(cum_stats)
            out_dict[date].append(last_10)
            i += 1
    except Exception as e:
        driver.quit()
        print(e)
        return info[2],out_dict,missing
    print(f'Data for: {info[2]} complete, Captured: {len(out_dict)}, Missing: {len(missing)}')
    driver.quit()
    return info[2],out_dict,missing

def get_stats_multi_win(info):
    '''
    Go through list of dates and populate ALL_TEAMS_CONC
    dates: list of dates similar to `Tue Oct 29 2013`
    base_url: base url of NBA stats page for this team
    abr: team abbreviation
    RETURNS: dict mapping date -> [cum_stats, last10]
    '''
    dates = info[0]
    base_url = info[1]
    options = Options()
    chrome_ops = webdriver.ChromeOptions()
    chrome_ops.add_argument("--headless=new")
    options.headless = True
    driver = webdriver.Chrome(options=options) # requires appropriate webdriver in PATH variables, or pass path to file as arg `executable_path=`
    out_dict = defaultdict(list)
    missing = []

    try:
        for i in range(9,len(dates),10): # open 5 tabs in one run
            date_list = []
            k = 0
            for j in range(i-9,i+1):
                curr_date = dates[j].split()
                date_list.append(dates[j])
                team_stats = base_url + f'&DateTo={MONTH_TO_NUM[curr_date[1]]}%2F{curr_date[2]}%2F{curr_date[3]}&Split=lastn'
                if j < 82: #TODO: remove this
                    team_stats += '&SeasonType=Regular+Season'
                else:
                    team_stats += '&SeasonType=Playoffs'
                driver.get(team_stats)
                k += 1
                if k < 10:
                    driver.execute_script("window.open()")
                    driver.switch_to.window(driver.window_handles[k])
            
            window_list = driver.window_handles
            k = 0
            # load JS
            # time.sleep(2)
            for window in window_list:
                date = date_list[k]
                k += 1
                driver.switch_to.window(window)
                html = driver.page_source

                soup = BeautifulSoup(html, 'html.parser')
                results = soup.find_all('table')

                try:
                    # 24 stats total from i = 1 to 24, only care about 3 - 24
                    # GP	MIN	PTS	W	L	WIN%	FGM	FGA	FG%	3PM	3PA	3P%	FTM	FTA	FT%	OREB	DREB	REB	AST	TOV	STL	BLK	PF	+/-
                    cumalitive = results[2].find_all('td') # cum stats to this point
                    cum_stats = []
                    for e in cumalitive[3:]: # only care about last ten games?
                        cum_stats.append(float(e.text))

                    per_n = results[3].find_all('tr') # every 10 games ranges
                    last_10 = []

                    # if last 10 range includes at least 5 games, use those, else use previous 10 range
                    if float(per_n[-1].find_all('td')[1].text) >= 5 or i < 10 or len(per_n) < 2:
                        for e in per_n[-1].find_all('td')[3:]:
                            last_10.append(float(e.text))
                        
                    else:
                        for e in per_n[-2].find_all('td')[3:]:
                            last_10.append(float(e.text))

                    out_dict[date].append(cum_stats)
                    out_dict[date].append(last_10)
                except:
                    missing.append(date)
                if len(driver.window_handles) > 1:
                    driver.close()
        
        date_list = []
        k = 0
        remaining = len(dates) % 10
        if remaining > 0:
            for i in range(len(dates) - remaining, len(dates)):
                curr_date = dates[i].split()
                date_list.append(dates[i])
                team_stats = base_url + f'&DateTo={MONTH_TO_NUM[curr_date[1]]}%2F{curr_date[2]}%2F{curr_date[3]}&Split=lastn'
                if i < 82: #TODO: remove this
                    team_stats += '&SeasonType=Regular+Season'
                else:
                    team_stats += '&SeasonType=Playoffs'
                driver.get(team_stats)
                k += 1
                if k < remaining:
                    driver.execute_script("window.open()")
                    driver.switch_to.window(driver.window_handles[k])

            
            window_list = driver.window_handles
            k = 0
            # load JS
            # time.sleep(2)
            for window in window_list:
                date = date_list[k]
                k += 1
                driver.switch_to.window(window)
                html = driver.page_source

                soup = BeautifulSoup(html, 'html.parser')
                results = soup.find_all('table')

                try:
                    # 24 stats total from i = 1 to 24, only care about 3 - 24
                    # GP	MIN	PTS	W	L	WIN%	FGM	FGA	FG%	3PM	3PA	3P%	FTM	FTA	FT%	OREB	DREB	REB	AST	TOV	STL	BLK	PF	+/-
                    cumalitive = results[2].find_all('td') # cum stats to this point
                    cum_stats = []
                    for e in cumalitive[3:]: # only care about last ten games?
                        cum_stats.append(float(e.text))

                    per_n = results[3].find_all('tr') # every 10 games ranges
                    last_10 = []

                    # if last 10 range includes at least 5 games, use those, else use previous 10 range
                    if float(per_n[-1].find_all('td')[1].text) >= 5 or i < 10 or len(per_n) < 2:
                        for e in per_n[-1].find_all('td')[3:]:
                            last_10.append(float(e.text))
                        
                    else:
                        for e in per_n[-2].find_all('td')[3:]:
                            last_10.append(float(e.text))

                    out_dict[date].append(cum_stats)
                    out_dict[date].append(last_10)
                except:
                    missing.append(date)
                if len(driver.window_handles) > 1:
                    driver.close()

    except Exception as e:
        driver.quit()
        print(f'Data for: {info[2]} complete, Captured: {len(out_dict)}, Missing: {len(missing)}, Error: {e}')
        return info[2],out_dict,missing
    
    print(f'Data for: {info[2]} complete, Captured: {len(out_dict)}, Missing: {len(missing)}')
    driver.quit()
    return info[2],out_dict,missing

# import multiprocessing
def one_szn_par(games_df,szn):
    start_date = games_df['date'][0].split()
    teams = []

    # Generate list of URLS for each team
    for team, abr in TEAM_NAME_TO_ABR.items():
        team_games = games_df.loc[(games_df['away'].str.upper() == team) | (games_df['home'].str.upper() == team)]
        dates = team_games['date'].to_list()
        url = TEAM_ABR_TO_URL[abr] + f'/traditional?Season={szn}&DateFrom={MONTH_TO_NUM[start_date[1]]}%2F{start_date[2]}%2F{start_date[3]}'
        teams.append([dates,url,abr])
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(get_stats_multi_win,teams))

    return results

import datetime
# TODO: add batch size parameter/fix pooling 
def get_on_off(info):
    '''
    Gets cumulative advanced stats for a given season, NOT INCLUDING LAST 10
    input: [dates,url,abr,reg_szn_end,play_in_end]
    RETURNS: dict mapping date -> [abr,adv_stats]
    '''
    dates = info[0]
    base_url = info[1]
    abr = info[2]
    reg_szn_end = info[3]
    play_in_end = info[4]
    reg_szn_end_dt = datetime.datetime(int(reg_szn_end[3]),int(reg_szn_end[1]),int(reg_szn_end[2]))
    if len(play_in_end) > 0:
        play_in_end_dt = datetime.datetime(int(play_in_end[3]),int(play_in_end[1]),int(play_in_end[2]))

    options = Options()
    driver = webdriver.Chrome(options=options) # requires appropriate webdriver in PATH variables, or pass path to file as arg `executable_path=`
    # teams_dict = info[2]
    out_dict = defaultdict(list) # map date -> [abr,adv_stats]
    missing = []

    try:
        for i in range(9,len(dates),10): # change range step to control how many tabs are opened in a batch, current set to 5
            date_list = []
            k = 0
            for j in range(i-9,i+1):
                curr_date = dates[j].split()
                date_list.append(dates[j])
                team_stats = base_url + f'&DateTo={MONTH_TO_NUM[curr_date[1]]}%2F{curr_date[2]}%2F{curr_date[3]}'

                curr_date_dt = datetime.datetime(int(curr_date[3]),int(MONTH_TO_NUM[curr_date[1]]),int(curr_date[2]))

                if curr_date_dt <= reg_szn_end_dt:
                    team_stats += '&SeasonType=Regular+Season'
                elif len(play_in_end) > 0 and curr_date_dt <= play_in_end_dt:
                    team_stats += '&SeasonType=PlayIn'
                else:
                    team_stats += '&SeasonType=Playoffs'

                driver.get(team_stats)
                k += 1
                if k < 10: # change range step to control how many tabs are opened in a batch, current set to 10
                    driver.execute_script("window.open()")
                    driver.switch_to.window(driver.window_handles[k])
            
            window_list = driver.window_handles
            k = 0
            # load JS
            time.sleep(2)
            for window in window_list:
                date = date_list[k]
                k += 1
                driver.switch_to.window(window)
                html = driver.page_source

                soup = BeautifulSoup(html, 'html.parser')
                results = soup.find_all('table')

                try:
                    overall = results[2].find_all('tr')[1].find_all('td')
                    over_stats = []
                    on_dict = defaultdict(list)
                    off_dict = defaultdict(list)

                    for over in overall[1:]:
                        over_stats.append(float(over.text))

                    out_dict[date].append(over_stats)

                    # NAME	GP	MIN	OFFRTG	DEFRTG	NETRTG	AST%	AST/TO	AST RATIO	OREB%	DREB%	REB%	TOV%	EFG%	TS%	PACE	PIE
                    on_court = results[3].find_all('tr')[1:]
                    for player in on_court:
                        on_stats = []
                        player_stats = player.find_all('td')
                        name_strip = player_stats[0].text
                        d = ''.join(name_strip.split())
                        full_name = ' '.join(d.split(',')[::-1]).upper()
                        for e in player_stats[1:]:
                            on_stats.append(float(e.text))
                        on_dict[full_name] = on_stats

                    out_dict[date].append(on_dict)

                    off_court = results[4].find_all('tr')[1:]
                    for player in off_court:
                        off_stats = []
                        player_stats = player.find_all('td')
                        name_strip = player_stats[0].text
                        d = ''.join(name_strip.split())
                        full_name = ' '.join(d.split(',')[::-1]).upper()
                        for e in player_stats[1:]:
                            off_stats.append(float(e.text))
                        off_dict[full_name] = off_stats

                    out_dict[date].append(off_dict)

                except Exception as e:
                    print(f'Exception: {e} for team: {info[2]} on link: {team_stats}')
                    missing.append(date)
                if len(driver.window_handles) > 1:
                    driver.close()
        
        date_list = []
        k = 0
        remaining = len(dates) % 10
        if remaining > 0:
            for i in range(len(dates) - remaining, len(dates)):
                curr_date = dates[i].split()
                date_list.append(dates[i])
                team_stats = base_url + f'&DateTo={MONTH_TO_NUM[curr_date[1]]}%2F{curr_date[2]}%2F{curr_date[3]}'

                curr_date_dt = datetime.datetime(int(curr_date[3]),int(MONTH_TO_NUM[curr_date[1]]),int(curr_date[2]))

                if curr_date_dt <= reg_szn_end_dt:
                    team_stats += '&SeasonType=Regular+Season'
                elif len(play_in_end) > 0 and curr_date_dt <= play_in_end_dt:
                    team_stats += '&SeasonType=PlayIn'
                else:
                    team_stats += '&SeasonType=Playoffs'

                driver.get(team_stats)
                k += 1
                if k < remaining:
                    driver.execute_script("window.open()")
                    driver.switch_to.window(driver.window_handles[k])

            
            window_list = driver.window_handles
            k = 0
            # load JS
            time.sleep(2)
            for window in window_list:
                date = date_list[k]
                k += 1
                driver.switch_to.window(window)
                html = driver.page_source

                soup = BeautifulSoup(html, 'html.parser')
                results = soup.find_all('table')

                try:
                    overall = results[2].find_all('tr')[1].find_all('td')
                    over_stats = []
                    on_dict = defaultdict(list)
                    off_dict = defaultdict(list)

                    for over in overall[1:]:
                        over_stats.append(float(over.text))

                    out_dict[date].append(over_stats)

                    # NAME	GP	MIN	OFFRTG	DEFRTG	NETRTG	AST%	AST/TO	AST RATIO	OREB%	DREB%	REB%	TOV%	EFG%	TS%	PACE	PIE
                    on_court = results[3].find_all('tr')[1:]
                    for player in on_court:
                        on_stats = []
                        player_stats = player.find_all('td')
                        name_strip = player_stats[0].text
                        d = ''.join(name_strip.split())
                        full_name = ' '.join(d.split(',')[::-1]).upper()
                        for e in player_stats[1:]:
                            on_stats.append(float(e.text))
                        on_dict[full_name] = on_stats

                    out_dict[date].append(on_dict)

                    off_court = results[4].find_all('tr')[1:]
                    for player in off_court:
                        off_stats = []
                        player_stats = player.find_all('td')
                        name_strip = player_stats[0].text
                        d = ''.join(name_strip.split())
                        full_name = ' '.join(d.split(',')[::-1]).upper()
                        for e in player_stats[1:]:
                            off_stats.append(float(e.text))
                        off_dict[full_name] = off_stats

                    out_dict[date].append(off_dict)

                except Exception as e:
                    print(f'Exception: {e} for team: {abr} on link: {team_stats}')
                    missing.append(date)
                if len(driver.window_handles) > 1:
                    driver.close()

    except Exception as e:
        driver.quit()
        print(f'Team: {info[2]}, Captured: {len(out_dict)}, Missing: {len(missing)}, Error: {e}')
        return info[2],out_dict,missing
    
    print(f'Team: {info[2]}, Captured: {len(out_dict)}, Missing: {len(missing)}')
    driver.quit()
    return info[2],out_dict,missing

import concurrent
import concurrent.futures

def pop_on_off(games_df,szn,reg_szn,play_in):
    start_date = games_df['date'][0].split()
    teams = []

    # Generate list of URLS for each team
    for team, abr in TEAM_NAME_TO_ABR.items():
        team_games = games_df.loc[(games_df['away'].str.upper() == team) | (games_df['home'].str.upper() == team)]
        dates = team_games['date'].to_list()
        url = TEAM_ABR_TO_URL[abr] + f'/onoffcourt-advanced?Season={szn}&DateFrom={MONTH_TO_NUM[start_date[1]]}%2F{start_date[2]}%2F{start_date[3]}'
        teams.append([dates,url,abr,reg_szn,play_in]) #regszn end, playin end
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(get_on_off,teams))

    return results

if __name__ == '__main__':
    df_list = []
    szn_list = []
    reg_szn_list = []
    play_in_list = []

    games_df_2023 = pd.read_csv('NBA/games/2022-2023_season_inj.csv',sep=',',names=['date','away','away_pt','home','home_pt'])
    reg_szn_end_2023 = ["","04","9","2023"]
    play_in_end_2023 = ["","04","14","2023"]

    res = pop_on_off(games_df_2023,'2022-23',reg_szn_end_2023,play_in_end_2023)
    on_off_dict = dict.fromkeys(TEAM_ABR_TO_URL.keys())

    for r in res:
        on_off_dict[r[0]] = r[1]
        on_off_dict[r[0]]["MISSED"] = r[2]
        
    with open('NBA/conc_on_off/2022-2023_conc_stats.pkl', 'wb') as f:
        pickle.dump(on_off_dict,f)

    games_df_2022 = pd.read_csv('NBA/games/2021-2022_season_inj.csv',sep=',',names=['date','away','away_pt','home','home_pt'])
    reg_szn_end_2022 = ["","04","10","2022"]
    play_in_end_2022 = ["","04","15","2022"]

    res = pop_on_off(games_df_2022,'2021-22',reg_szn_end_2022,play_in_end_2022)
    on_off_dict = dict.fromkeys(TEAM_ABR_TO_URL.keys())

    for r in res:
        on_off_dict[r[0]] = r[1]
        on_off_dict[r[0]]["MISSED"] = r[2]
        
    with open('NBA/conc_on_off/2021-2022_conc_stats.pkl', 'wb') as f:
        pickle.dump(on_off_dict,f)

    games_df_2021 = pd.read_csv('NBA/games/2020-2021_season_inj.csv',sep=',',names=['date','away','away_pt','home','home_pt'])
    reg_szn_end_2021 = ["","05","16","2021"]
    play_in_end_2021 = ["","05","21","2021"]

    res = pop_on_off(games_df_2021,'2020-21',reg_szn_end_2021,play_in_end_2021)
    on_off_dict = dict.fromkeys(TEAM_ABR_TO_URL.keys())

    for r in res:
        on_off_dict[r[0]] = r[1]
        on_off_dict[r[0]]["MISSED"] = r[2]
        
    with open('NBA/conc_on_off/2020-2021_conc_stats.pkl', 'wb') as f:
        pickle.dump(on_off_dict,f)

    games_df_2020 = pd.read_csv('NBA/games/2019-2020_season_inj.csv',sep=',',names=['date','away','away_pt','home','home_pt'])
    reg_szn_end_2020 = ["","08","14","2020"]
    play_in_end_2020 = ["","08","15","2020"]

    res = pop_on_off(games_df_2020,'2019-20',reg_szn_end_2020,play_in_end_2020)
    on_off_dict = dict.fromkeys(TEAM_ABR_TO_URL.keys())

    for r in res:
        on_off_dict[r[0]] = r[1]
        on_off_dict[r[0]]["MISSED"] = r[2]
        
    with open('NBA/conc_on_off/2019-2020_conc_stats.pkl', 'wb') as f:
        pickle.dump(on_off_dict,f)

    games_df_2019 = pd.read_csv('NBA/games/2018-2019_season_inj.csv',sep=',',names=['date','away','away_pt','home','home_pt'])
    reg_szn_end_2019 = ["","04","10","2019"]
    play_in_end_2019 = []

    res = pop_on_off(games_df_2019,'2018-19',reg_szn_end_2019,play_in_end_2019)
    on_off_dict = dict.fromkeys(TEAM_ABR_TO_URL.keys())

    for r in res:
        on_off_dict[r[0]] = r[1]
        on_off_dict[r[0]]["MISSED"] = r[2]
        
    with open('NBA/conc_on_off/2018-2019_conc_stats.pkl', 'wb') as f:
        pickle.dump(on_off_dict,f)

    games_df_2018 = pd.read_csv('NBA/games/2017-2018_season_inj.csv',sep=',',names=['date','away','away_pt','home','home_pt'])
    reg_szn_end_2018 = ["","04","11","2018"]
    play_in_end_2018 = []

    res = pop_on_off(games_df_2018,'2017-18',reg_szn_end_2018,play_in_end_2018)
    on_off_dict = dict.fromkeys(TEAM_ABR_TO_URL.keys())

    for r in res:
        on_off_dict[r[0]] = r[1]
        on_off_dict[r[0]]["MISSED"] = r[2]
        
    with open('NBA/conc_on_off/2017-2018_conc_stats.pkl', 'wb') as f:
        pickle.dump(on_off_dict,f)

    games_df_2017 = pd.read_csv('NBA/games/2016-2017_season_inj.csv',sep=',',names=['date','away','away_pt','home','home_pt'])
    reg_szn_end_2017 = ["","04","12","2017"]
    play_in_end_2017 = []

    res = pop_on_off(games_df_2017,'2016-17',reg_szn_end_2017,play_in_end_2017)
    on_off_dict = dict.fromkeys(TEAM_ABR_TO_URL.keys())

    for r in res:
        on_off_dict[r[0]] = r[1]
        on_off_dict[r[0]]["MISSED"] = r[2]
        
    with open('NBA/conc_on_off/2016-2017_conc_stats.pkl', 'wb') as f:
        pickle.dump(on_off_dict,f)