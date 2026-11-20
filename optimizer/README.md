# 1-month sanity run (fast)
python run_phase2_baseline.py \
  --data ../path/to/your_shaved_ice.csv \
  --start 2021-02-01 \
  --end 2021-03-01 \
  --discount 0.40

# Full dataset run
python run_phase2_baseline.py \
  --data ../path/to/your_shaved_ice.csv \
  --discount 0.40