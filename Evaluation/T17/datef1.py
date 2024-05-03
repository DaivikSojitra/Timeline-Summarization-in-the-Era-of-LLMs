import pandas as pd
from datetime import datetime
from evaluate import load
from statistics import mean

bleu = load("bleu")
rouge = load('rouge')

def evaluate_dates(pred, ground_truth):
    pred_dates = pred
    ref_dates = ground_truth
    shared = pred_dates.intersection(ref_dates)
    n_shared = len(shared)
    n_pred = len(pred_dates)
    n_ref = len(ref_dates)
    prec = n_shared / n_pred
    rec = n_shared / n_ref
    if prec + rec == 0:
        f_score = 0
    else:
        f_score = 2 * prec * rec / (prec + rec)
    return {
        'precision': prec,
        'recall': rec,
        'f_score': f_score,
    }


storiesResponseList=[]
storiesSummaries=[]
corrections=[]
finalsummaries=[]
blue = []
Rouge1 = []
Rouge2 = []
RougeL = []
RougeLsum = []
prec = []
rec = []
f_score = []

stories = ["bpoil", "egypt", "finan", "h1n1", "haiti", "iraq", "libya", "mj", "syria"]
datasetpath = "/home/sojitra_2211mc15/Daivik/KG-LLM/T17/LongL/KG-T17/output-"

import csv

for k in range(len(stories)):
    path = datasetpath + stories[k] + ".csv";
    df = pd.read_csv(path,header=None)
    # df = df.iloc[1:]
    print(df.head())
    print(df.values.tolist()[0])

    summariesdates = []
    summaries = []

    for i in range(len(df.values.tolist())):
        summariesdates.append(df.values.tolist()[i][0])
        summaries.append(df.values.tolist()[i][1])

    dateset = set()
    t = ""
    for line in range(len(summaries)):

        dateset.add(summariesdates[line])
        
        t = t + " " + str(summaries[line])
        # Strip the newline character and append the line to the list
        # file_contents.append(line.strip())


    # Print or process the file contents as needed
    print(dateset)
    # print(t)
    # file_contents.append(t)

    opath = "/home/sojitra_2211mc15/Daivik/MDS/t17/" + stories[k] + "/outputTimelinesTime.csv"
    df = pd.read_csv(opath,header=None)
    df = df.iloc[1:]

    refdate = set()

    for line in range(len(df.values.tolist())):
        data_str = df.values.tolist()[line][0]
        # x = strdate.split(",")
        # Remove the outer single quotes and split the string into a list
        data_str = data_str[1:-1]  # Remove the outer single quotes
        date_list = [date.strip() for date in data_str.split(',')]

        # Iterate through the list and print each date
        for date_str in date_list:
            date_str = date_str[1:11]
            refdate.add(date_str)

    dateres = evaluate_dates(dateset, refdate)
    print("Date results: ", dateres)

    prec.append(dateres['precision'])
    rec.append(dateres['recall'])
    f_score.append(dateres['f_score'])

    filefinal  = "flan-final-result.txt"
    f1 = csv.writer(open(filefinal, "a+"))
    headsen = "\n################# " + stories[k] + " ####################\n"

    f1.writerow(["Date results: ", dateres])
    f1.writerow(["Final precison Score : ", mean(prec)])
    f1.writerow(["Final recall Score : ", mean(rec)])
    f1.writerow(["Final f1 Score : ", mean(f_score)])

    print("Final precison Score : ", mean(prec))
    print("Final recall Score : ", mean(rec))
    print("Final f1 Score : ", mean(f_score))
