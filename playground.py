# NBA Import
from espn_api.basketball import League
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# private league with cookies
league = League(league_id=1338283269, year=2023, espn_s2=os.environ['ESPN_S2'], swid=os.environ['SWID'], fetch_league=True)
teams = league.teams
weights = {'PTS': 1, '3PTM': 1, 'FGA': -1, 'FGM': 2, 'FTA': -1, 'FTM': 1, 'REB': 1, 'AST': 2, 'STL': 4, 'BLK': 4, 'TO': -2}
# my_team = teams[0]
# players = my_team.roster
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