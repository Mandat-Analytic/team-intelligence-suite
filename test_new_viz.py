import radar_chart
import pandas as pd
import matplotlib.pyplot as plt

print("Testing new radar_chart functions...")

# 1. Test create_phase_pizza
scores = {
    "Build Up": 85,
    "Progression": 72, 
    "Transition": 45,
    "Output": 90,
    "Set Pieces": 60
}

try:
    fig = radar_chart.create_phase_pizza(scores, "Test Chart")
    print("✅ create_phase_pizza passed")
except Exception as e:
    print(f"❌ create_phase_pizza failed: {e}")

# 2. Test calculate_player_percentile
data = {
    'Player': ['A', 'B', 'C', 'D', 'E'],
    'Position': ['CF', 'CF', 'CF', 'CF', 'GK'],
    'Goals': [10, 5, 2, 8, 0],
    'Minutes played': [1000, 1000, 1000, 1000, 1000]
}
df = pd.DataFrame(data)

# Test percentile for player A (10 goals) - should be 100th percentile (or close)
try:
    p = radar_chart.calculate_player_percentile(10, 'Goals', 'Forward', df)
    print(f"✅ calculate_player_percentile passed (Result: {p:.1f})")
except Exception as e:
    print(f"❌ calculate_player_percentile failed: {e}")
