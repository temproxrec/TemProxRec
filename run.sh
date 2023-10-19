python main.py --dataset_code='books' --min_uc=20 --min_sc=30 --hidden_units=128 --dropout=0.3 --clip_time=256 --weight_decay=0 --interval=60 --temperature=0.05 --lamb=0.3
python main.py --dataset_code='video' --hidden_units=128 --dropout=0.3 --clip_time=256 --weight_decay=0.00001 --interval=7 --temperature=0.05 --lamb=0.4
python main.py --dataset_code='steam' --min_uc=10 --hidden_units=128 --dropout=0.3 --clip_time=512  --weight_decay=0 --interval=30 --temperature=0.05 --lamb=0.3
python main.py --dataset_code='beauty' --hidden_units=32 --dropout=0.1 --clip_time=128 --weight_decay=0 --interval=60 --temperature=0.05 --lamb=0.3
