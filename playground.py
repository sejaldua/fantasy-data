# NBA Import
from espn_api.basketball import League
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np

# private league with cookies
league = League(league_id=1338283269, year=2023, espn_s2=os.environ['ESPN_S2'], swid=os.environ['SWID'], fetch_league=True)
teams = league.teams
weights = {'PTS': 1, '3PTM': 1, 'FGA': -1, 'FGM': 2, 'FTA': -1, 'FTM': 1, 'REB': 1, 'AST': 2, 'STL': 4, 'BLK': 4, 'TO': -2}


# PART 1 - predicting stat importances at player level
columns = ['Name', 'FTA', 'PTS', '3PTM', 'BLK', 'STL', 'AST', 'REB', 'TO', 'FGM', 'FGA', 'FTM', 'Fantasy Points']
df = pd.DataFrame([], columns=columns)
player_data = []
num_weeks = 8
for week in range(num_weeks):
    matchups = league.box_scores(matchup_period=week+1)
    for matchup in matchups:
        for player in [*matchup.home_lineup, *matchup.away_lineup]:
            breakdown = player.points_breakdown
            raw_player_stats = {k: v / weights[k] for k, v in breakdown.items()}
            data = [[player.name, *raw_player_stats.values(), player.points]]
            new_df = pd.DataFrame(data, columns=columns)
            df = pd.concat([df, new_df], axis=0)
            
df = df.set_index('Name')
X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)
# Make predictions using the testing set
y_pred = regr.predict(X_test)
# the weights that allow the model to predict the player's total fantasy points
print({k: round(v, 2) for k, v in zip(list(df.columns[:-1]), list(regr.coef_))})


# PART 2: predicting stat importances at team level
columns = ['FTA', 'PTS', '3PTM', 'BLK', 'STL', 'AST', 'REB', 'TO', 'FGM', 'FGA', 'FTM', 'GP', 'Win Indicator']
df = pd.DataFrame([], columns=columns)
player_data = []
num_weeks = 8
for week in range(num_weeks):
    matchups = league.box_scores(matchup_period=week+1)
    for matchup in matchups:
        if matchup.winner != 'UNDECIDED':
            teams = ['HOME', 'AWAY']
            for i, team in enumerate([matchup.home_lineup, matchup.away_lineup]):
                team_stats = {k: 0 for k in columns[:-1]}
                for player in team:
                    breakdown = player.points_breakdown
                    raw_player_stats = {k: v / weights[k] for k, v in breakdown.items()}
                    raw_player_stats['GP'] = player.game_played / 100
                    team_stats = {k: v + raw_player_stats[k] for k, v in team_stats.items()}
                data = list(team_stats.values())
                winner = 1 if matchup.winner == teams[i] else 0
                data.append(winner)
                new_df = pd.DataFrame([data], columns=columns)
                df = pd.concat([df, new_df], axis=0)
              
df = df.reset_index(drop=True)
print(df)
X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape)

# Create linear regression object
regr = linear_model.LogisticRegression(max_iter=1000, warm_start=True, solver='liblinear')
# Train the model using the training sets
regr.fit(X_train, y_train)
# Make predictions using the testing set
# y_pred = regr.predict(X_test)
# the weights that allow the model to predict the player's total fantasy points
odds = np.exp(regr.coef_[0])
coef_df = pd.DataFrame(odds, 
             df.columns[:-1], 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)
print(coef_df)


# PART 3: predicting stat importances at matchup level
columns = ['FTA', 'PTS', '3PTM', 'BLK', 'STL', 'AST', 'REB', 'TO', 'FGM', 'FGA', 'FTM', 'GP', 'Home Team Wins']
df = pd.DataFrame([], columns=columns)
player_data = []
num_weeks = 8
for week in range(num_weeks):
    matchups = league.box_scores(matchup_period=week+1)
    for matchup in matchups:
        if matchup.winner != 'UNDECIDED':
            arrs = []
            for team in [matchup.home_lineup, matchup.away_lineup]:
                team_stats = {k: 0 for k in columns[:-1]}
                for player in team:
                    breakdown = player.points_breakdown
                    raw_player_stats = {k: v / weights[k] for k, v in breakdown.items()}
                    raw_player_stats['GP'] = player.game_played / 100
                    team_stats = {k: v + raw_player_stats[k] for k, v in team_stats.items()}
                data = list(team_stats.values())
                arrs.append(data)
            diff = [x - y for x, y in zip(arrs[0], arrs[1])]
            winner = 1 if matchup.winner == 'HOME' else 0
            diff.append(winner)
            new_df = pd.DataFrame([diff], columns=columns)
            df = pd.concat([df, new_df], axis=0)
              
df = df.reset_index(drop=True)
print(df)
X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape)

# Create linear regression object
regr = linear_model.LogisticRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)
# Make predictions using the testing set
# y_pred = regr.predict(X_test)
# the weights that allow the model to predict the player's total fantasy points
odds = np.exp(regr.coef_[0])
coef_df = pd.DataFrame(odds, 
             df.columns[:-1], 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)
print(coef_df)