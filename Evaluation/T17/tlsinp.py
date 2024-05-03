import pandas as pd
import csv

stories = ["bpoil", "egypt", "finan", "h1n1", "haiti", "iraq", "libya", "mj", "syria"]

for i in range(len(stories)):
    path = "/Evaluation/T17/Gold/outputTimelines"+ stories[i] + ".csv"

    df = pd.read_csv(path,header=None)
    # df = df.iloc[1:]
    print(df.head())
    # print(df.values.tolist()[0])

    wrpath = "/Evaluation/T17/Gold/outputTimelines"+ stories[i] + ".txt"
    f = csv.writer(open(wrpath, "w+"))

    for i in range(len(df.values.tolist())):
        f.writerow([df.values.tolist()[i][0]])
        f.writerow([df.values.tolist()[i][1]])
        f.writerow(["--------------------------------"])
