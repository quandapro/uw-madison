import argparse
import numpy as np
import pandas as pd
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, help="Input foler")
args = parser.parse_args()

csv_files = glob.glob(args.folder + "/*.csv")

val_dice = []
for csv in csv_files:
    df = pd.read_csv(csv)
    val_dice.append(np.max(df["val_Dice_Coef"]))

print(f"Parsing path {args.folder}")
print(val_dice)
print(f"{np.mean(val_dice)} +- {np.std(val_dice)}")