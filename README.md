# FantasyFootball2025
Fantasy Football 2025 Draft advanced statistics rankings with live draft aid

How to run (5 mins)

Make a fresh venv and install deps:

pip install pandas numpy requests

Build your board:

python fantasy_draft_helper.py build --out cheat_sheet.csv --since 2022 --through 2024 --expect_games 16.5
# (optional) add ADP to break ties:
python fantasy_draft_helper.py build --out cheat_sheet.csv --adp_csv my_adp.csv
# my_adp.csv columns: player, adp_overall


During the draft, get pick recommendations that respect your current roster + already drafted names:

python fantasy_draft_helper.py recommend \
  --cheatsheet cheat_sheet.csv \
  --team QB=0,RB=0,WR=0,TE=0 \
  --drafted "Justin Jefferson, Christian McCaffrey" \
  --topk 10


You’ll see a ranked list with position, tier, projected points, VOR, and a roster-need-adjusted score.