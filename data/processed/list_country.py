import pandas as pd

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "matches_with_features.csv"

df = pd.read_csv(csv_path)
countries = sorted(
    set(df["team_a"].dropna()) |
    set(df["team_b"].dropna())
)

print(countries)