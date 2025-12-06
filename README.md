# Team Intelligence Suite

A comprehensive football team performance analytics application built with Streamlit.

## Features

- **Performance Tree View**: Hierarchical visualization of team performance metrics
  - Game Model scoring (0-100 scale)
  - In Possession and Out of Possession phases
  - Detailed metrics for each tactical phase
  - Visual indicators (color-coded borders and arrows)

- **Deep Analytics**: 
  - Random Forest-based trend forecasting
  - Model accuracy metrics (R², RMSE, MAE)
  - Correlation matrix visualization

- **Player Intelligence**:
  - Position-specific radar charts (PyPizza)
  - Player comparison (side-by-side)
  - Team player rankings
  - League-wide percentile comparisons

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run Tree_Matrices.py
```

## Data Structure

The application expects the following folder structure:
```
database/
├── Team Stats/
│   └── Team Stats {Team Name}.xlsx
└── Player Stats/
    └── Serbia Super Liga 25_26.xlsx
```

## Technologies

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, mplsoccer
- **ML**: scikit-learn (Random Forest)

## Credits

Built for advanced football analytics and team performance intelligence.
