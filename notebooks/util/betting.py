# import statements
import numpy as np
from bs4 import BeautifulSoup, Comment
from collections import defaultdict
import pandas as pd
from tqdm.auto import tqdm
import concurrent
import concurrent.futures
import pyro
from pyro.infer import Predictive
import torch
from sklearn.metrics import classification_report,confusion_matrix,make_scorer

def calc_implied(home_line,away_line,log_home,log_away):
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

    return diff_home,home_ratio,diff_away,away_ratio

def kelly(home_pred,away_pred,home_lines,away_lines,max_bet=100,diff_thresh=0.05,diff_cap=0.25,log_probs=True):
    '''
    Applies kelly critereon based on features and moneyline data

    `home_pred`: Prediction from MLP for home team
    `away_pred`: Prediction from MLP for away team
    `home_line`: Moneyline for home team
    `away_line`: Moneyline for away team
    `diff_thresh`: Minimum difference between prediction and implied odds
    `diff_cap`: Maximum difference between prediciton and implied odds
    `log_probs`: Boolean flag to use log probabilities instead of raw
    '''
    bet_amount = 0
    to_win = 0

    if log_probs:
        log_home = (home_pred - (home_pred * away_pred)) / (home_pred + away_pred - (2*home_pred*away_pred))
        log_away = (away_pred - (home_pred * away_pred)) / (home_pred + away_pred - (2*home_pred*away_pred))
    else:
        log_home = home_pred
        log_away = away_pred

    max_diff = -float("inf")
    home_line = None
    away_line = None
    best_diff_home = -float("inf")
    best_diff_away = -float("inf")

    # find best lines for all given lines
    for i in range(len(home_lines)):
        # calculate ratio and implied
        diff_home,home_ratio,diff_away,away_ratio = calc_implied(home_lines[i],away_lines[i],log_home,log_away)
        
        if diff_home > max_diff:
            max_diff = diff_home
            home_line = home_lines[i]
            away_line = away_lines[i]
            best_diff_home = diff_home
            best_diff_away = diff_away

        elif diff_away > max_diff:
            max_diff = diff_home
            home_line = home_lines[i]
            away_line = away_lines[i]
            best_diff_home = diff_home
            best_diff_away = diff_away

    kelly_home = log_home - (log_away/home_ratio)
    kelly_away = log_away - (log_home/away_ratio)

    prob = 0

    # make bets, negative if away team bet
    if best_diff_home > best_diff_away and best_diff_home > diff_thresh and best_diff_home < diff_cap:
        bet_amount = (max_bet*kelly_home)
        if home_line < 0:
            to_win = bet_amount/((home_line*-1)/100)
        else:
            to_win = bet_amount/((home_line)/100)
        prob = home_pred

    
    elif best_diff_away > best_diff_home and best_diff_away > diff_thresh and best_diff_away < diff_cap:
        bet_amount = (max_bet*kelly_away)
        if away_line < 0:
            to_win = -1*bet_amount/((away_line*-1)/100)
        else:
            to_win = -1*bet_amount/((away_line)/100)
        prob = away_pred

    return bet_amount,to_win,prob

def BNN_kelly(preds,actual,money_lines,one_hot=False,diff_thresh=0.05,diff_cap=0.25,log_probs=True):
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

        to_bet,to_win,prob = kelly(home_pred,away_pred,home_ml,away_ml,diff_thresh=diff_thresh,diff_cap=diff_cap,log_probs=log_probs)
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
    `log_probs`: Boolean flag to use log probabilities instead of raw
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
              bet_samps_test,use_obs=True,use_ret=True,diff_thresh=0.05,diff_cap=0.25,
              categorical=True,verbose=True,log_probs=True):
    
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
    `verbose`: Boolean flag to print bets
    `log_probs`: Boolean flag to use log probabilities instead of raw

    Returns:
    obs_train,obs_test,ret_train,ret_test
    '''
    
    if use_obs == False and use_ret == False:
        print('ERROR: set "use_obs" or "use_ret" to True')
        return
    
    bet_samps_train_1d = [0 if j[0] < j[1] else 1 for j in bet_samps_train]
    bet_samps_test_1d = [0 if j[0] < j[1] else 1 for j in bet_samps_test]
    obs_train = -1
    obs_test = -1
    ret_train = -1
    ret_test = -1

    if verbose:
        print('PREDICTIONS ON 2022-2023 DATA (SEEN IN TRAINING)')
    if use_obs:
        if categorical:
            new_y_pred = train_preds['obs'].float().mean(axis=1).float().mean(axis=0)
        else:
            new_y_pred = train_preds['obs'].float().mean(axis=0)

        correct,guessed,team_bet,probs,amount,gained = BNN_kelly(new_y_pred,bet_samps_train_1d,bet_data_train[1:],one_hot=True,diff_thresh=diff_thresh,diff_cap=diff_cap,log_probs=log_probs)
        if verbose:
            print('Using OBS:')
            print(f'max confidence: {new_y_pred.max():.2f}')
            print(f'correct: {correct}')
            print(f'guessed: {guessed}')
            print(f'risked: {sum(amount)}')
            print(f'made: {sum(gained)}')
            print(f'ROI: {(sum(gained)/sum(amount)):.2f}\n')
        obs_train = (sum(gained)/sum(amount))

    
    if use_ret:
        new_y_pred = train_preds['_RETURN'].float().mean(axis=0)
        correct,guessed,team_bet,probs,amount,gained = BNN_kelly(new_y_pred,bet_samps_train_1d,bet_data_train[1:],one_hot=True,diff_thresh=diff_thresh,diff_cap=diff_cap,log_probs=log_probs)
        if verbose:
            print('Using RET:')
            print(f'max confidence: {new_y_pred.max():.2f}')
            print(f'correct: {correct}')
            print(f'guessed: {guessed}')
            print(f'risked: {sum(amount)}')
            print(f'made: {sum(gained)}')
            print(f'ROI: {(sum(gained)/sum(amount)):.2f}\n')
        ret_train = (sum(gained)/sum(amount))

    if verbose:
        print('PREDICTIONS ON 2023-2024 DATA (UNSEEN)')
    if use_obs:
        if categorical:
            new_y_pred = test_preds['obs'].float().mean(axis=1).float().mean(axis=0)
        else:
            new_y_pred = test_preds['obs'].float().mean(axis=0)
            
        correct,guessed,team_bet,probs,amount,gained = BNN_kelly(new_y_pred,bet_samps_test_1d,bet_data_test[1:],one_hot=True,diff_thresh=diff_thresh,diff_cap=diff_cap,log_probs=log_probs)
        if verbose:
            print('Using OBS:')
            print(f'max confidence: {new_y_pred.max():.2f}')
            print(f'correct: {correct}')
            print(f'guessed: {guessed}')
            print(f'risked: {sum(amount)}')
            print(f'made: {sum(gained)}')
            print(f'ROI: {(sum(gained)/sum(amount)):.2f}\n')
        obs_test = (sum(gained)/sum(amount))

    if use_ret:
        new_y_pred = test_preds['_RETURN'].float().mean(axis=0)
        correct,guessed,team_bet,probs,amount,gained = BNN_kelly(new_y_pred,bet_samps_test_1d,bet_data_test[1:],one_hot=True,diff_thresh=diff_thresh,diff_cap=diff_cap,log_probs=log_probs)
        if verbose:
            print('Using RET:')
            print(f'max confidence: {new_y_pred.max():.2f}')
            print(f'correct: {correct}')
            print(f'guessed: {guessed}')
            print(f'risked: {sum(amount)}')
            print(f'made: {sum(gained)}')
            print(f'ROI: {(sum(gained)/sum(amount)):.2f}\n')
        ret_test = (sum(gained)/sum(amount))

    return obs_train,obs_test,ret_train,ret_test