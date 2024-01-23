# import statements
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import time
import csv
from collections import defaultdict
import pickle
import random

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

    def __init__(self, start_year, end_year, team_stats=defaultdict(list), team_on_off=defaultdict(dict), features=None, samples=None):
        '''
        start_year: String representing start year of NBA season
        end_year: String representing end year of NBA season
        team_stats: Dict mapping team abr to stats
        team_on_off: Dict mapping team abr to dict of player names mapping to on-off stats
        features: List of unnormalized features for games representing the season
        samples: List of samples corresponding to features
        '''
        self.start_year = start_year
        self.end_year = end_year
        self.team_stats = team_stats
        self.team_on_off = team_on_off
        self.features = features
        self.samples = samples
    
    # get team stats for a given season from https://www.basketball-reference.com/teams/
    def get_team_stats(self,team_abr):
        '''
        team_abr: string abbreviation from basketball-reference
        end_year: string representing end year of season to get stats
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
    def calc_injury_impact(self,injured,home_abr,away_abr):

        home_injured = injured[home_abr]
        away_injured = injured[away_abr]
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

        return home_stats,away_stats


    def check_injured(self,box_score_page,home_abr,away_abr):
        '''
        Checks list of injured players for a given game and returns a dict mapping teams to injured player
        box_score_page: string representing URL for a given game
        home_team: string abbreviation of home team
        away_team: string abbreviation of away team
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

        return self.calc_injury_impact(players_dict,home_abr,away_abr)


    def generate_features(self,file_path):
        '''
        Returns lists containing features, samples
        file_path: path of CSV containing games for a season
        '''
        features = []
        samples = []
        #file_path = "old_games/{file_name}".format(file_name=file_name)
        line_count = 1
        fail_count = 0

        # construct data set, consisting of team misc stats as features and win/loss as samples
        with open(file_path, mode='r') as f:
            lines = csv.reader(f)
            for date,away_team,away_pt,home_team,home_pt in lines:
                try:
                    date_list = date.split()
                    month = self.MONTH_TO_NUM[date_list[1]]
                    day = date_list[2] if len(date_list[2]) == 2 else "0{day}".format(day=date_list[2])
                    year = date_list[3]
                    # TODO: change to 1d here?
                    results = [1,0] if away_pt > home_pt else [0,1]
                    
                    # get box score page
                    box_score_page = "https://www.basketball-reference.com/boxscores/{YEAR}{MO}{DA}0{HOME}.html".format(YEAR=year,MO=month,DA=day,HOME=self.TEAM_NAME_TO_ABR[home_team.upper()])
                    home_stats,away_stats = self.check_injured(box_score_page,self.TEAM_NAME_TO_ABR[home_team.upper()],self.TEAM_NAME_TO_ABR[away_team.upper()])
                    
                    line_count += 1

                    features.append(np.subtract(away_stats,home_stats))
                    samples.append(results)
                except Exception as e:
                    print(e)
                    #print(box_score_page)
                    print("Failed at line: {count} for file: {file_name} LINK: {link}".format(count=line_count,file_name=file_path,link=box_score_page))
                    fail_count += 1
                    time.sleep(random.randint(1, 5))
                    if fail_count > 5:
                        self.features = features
                        self.samples = samples
                        print("Too many failures, terminating feature generation")
                        return features,samples
                    continue
        self.features = features
        self.samples = samples
        return features,samples

    # populate team stats and on off stats for new season
    def pop_const_new(self):
        '''
        Populate team stats and on-off data
        '''
        self.pop_team_stats()

        with open('{start}-{end}_team_stats.pkl'.format(start=self.start_year,end=self.end_year), 'wb') as f:
            pickle.dump(self.team_stats, f)

        time.sleep(30)

        self.pop_team_on_off()

        with open('{start}-{end}_on_off.pkl'.format(start=self.start_year,end=self.end_year), 'wb') as f:
            pickle.dump(self.team_on_off, f)
    
    def save_data(self,save_path=''):
        '''
        Save features and samples to SAVE_PATH
        '''
        np.savetxt('{save_path}{start_year}-{end_year}_nba_features_inj.csv'.format(save_path=save_path,start_year=self.start_year,end_year=self.end_year), self.features, delimiter=',')
        np.savetxt('{save_path}{start_year}-{end_year}_nba_samples_inj.csv'.format(save_path=save_path,start_year=self.start_year,end_year=self.end_year), self.samples, delimiter=',')