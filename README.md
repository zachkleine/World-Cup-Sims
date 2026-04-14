# ⚽ World Cup Sims  
### Data-Driven FIFA World Cup Predictions & Monte Carlo Simulation Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-green)
![Last Commit](https://img.shields.io/github/last-commit/zachkleine/World-Cup-Sims)

---

## 🧠 What This Is

A simulation engine that predicts FIFA World Cup outcomes using a blend of:

- 📊 **Elo Ratings** (long-term team strength)
- 💰 **Betting Market Odds** (sharp, real-time signal)
- 📈 **Statistical Modeling** (expected goals + Poisson scoring)

Then runs **10,000+ Monte Carlo simulations** to estimate:

- 🏆 World Cup winners  
- 📊 Group advancement probabilities  
- ⚽ Goals scored / conceded  
- 🔁 Match-by-match outcomes  

---

## 🔥 Why This Project Exists

Most models rely on *either* Elo *or* betting odds.

This project combines both to:
- Capture **true team strength**
- Incorporate **market intelligence**
- Model **variance and upsets realistically**

> 💡 Insight: Betting markets are often the most efficient predictor — this model treats them as a **feature**, not a competitor.

---

## ⚙️ How It Works

### 1️⃣ Build Team Strength

- Convert American odds → implied probability  
- Remove bookmaker vig  
- Blend across sportsbooks  
- Combine with Elo ratings  
- (Optional) Add xG / squad value signals  

➡️ Output: `team_strengths.csv`

---

### 2️⃣ Simulate Matches

- Win probability derived from Elo difference  
- Expected goals calculated using scaling factor  
- Scores generated via **Poisson distribution**

---

### 3️⃣ Simulate Tournament

- Full World Cup structure (groups + knockout rounds)  
- 10,000+ simulations  
- Track outcomes across all runs  

---

## 🏗️ Project Structure

```bash
World-Cup-Sims/
│
├── data/                  # Raw & processed data (gitignored)
├── input/                 # Odds text files / manual inputs
│
├── scripts/
│   ├── build_strengths.py     # Blend Elo + odds
│   ├── odds_to_prob.py        # Devig + probability conversion
│
├── wcsims_full.py         # Main simulation engine
├── team_strengths.csv     # Final model input
├── requirements.txt
└── README.md