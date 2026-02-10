import pandas as pd
import os

player_file = r"c:\Users\USER\Desktop\Rembatz Analisis\Fiver Task\Tammo\Tree Matrices\database\Player Stats\Germany Frauen Bundesliga 25_26\Midfielder Germany Frauen Bundesliga 25_26.xlsx"

if os.path.exists(player_file):
    df = pd.read_excel(player_file, nrows=5)
    print("Columns in Player Stats:")
    for col in df.columns:
        print(f"- {col}")
else:
    print(f"File not found: {player_file}")
