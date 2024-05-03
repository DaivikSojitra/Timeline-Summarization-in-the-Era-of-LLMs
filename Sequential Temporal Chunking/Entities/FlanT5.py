import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from datetime import datetime
from evaluate import load
from statistics import mean
import csv

bleu = load("bleu")
rouge = load('rouge')

torch.cuda.set_device(0)
device = torch.device("cuda:0")  # Set the device

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
model = model.to(device) 
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

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

stories = ["Al_Gore", "Angela_Merkel", "Ariel_Sharon", "Arnold_Schwarzenegger", "Bashar_al-Assad", "Bill_Clinton", "Charles_Taylor", "Chris_Brown", "David_Beckham", "David_Bowie",  "Dilma_Rousseff", "Dmitry_Medvedev", "Dominique_Strauss-Kahn", "Edward_Snowden", "Ehud_Olmert", "Enron", "Hamid_Karzai", "Hassan_Rouhani", "Hu_Jintao", "Jacob_Zuma", "John_Boehner", "John_Kerry", "Julian_Assange", "Lance_Armstrong", "Mahmoud_Ahmadinejad", "Marco_Rubio", "Margaret_Thatcher", "Michael_Jackson", "Michelle_Obama", "Mitt_Romney", "Morgan_Tsvangirai", "Nawaz_Sharif", "Nelson_Mandela", "Osama_bin_Laden", "Oscar_Pistorius", "Phil_Spector", "Prince_William", "Robert_Mugabe", "Rupert_Murdoch", "Saddam_Hussein", "Sarah_Palin", "Silvio_Berlusconi", "Steve_Jobs", "Taliban", "Ted_Cruz", "Tiger_Woods", "WikiLeaks"]
datasetpath = "/Dataset/Entities/" 

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


for k in range(len(stories)):
    path = datasetpath + stories[k] + "/outputwithpub.csv"
    df = pd.read_csv(path,header=None)
    df = df.iloc[1:]
    print(df.head())
    print(df.values.tolist()[0])
    
    count = 1
    # Strips the newline character
    clusters = []
    dates = []
    summariesdates = []
    summaries = []
    temp = ""
    tempdates = []
    for line in range(len(df.values.tolist())):
        temp = temp + str(df.values.tolist()[line][1]) + " "
        tempdates.append(str(df.values.tolist()[line][0]))
        if count % 3 == 0:
            # Convert date strings to datetime objects
            date_objects = [datetime.strptime(date, "%Y-%m-%d") for date in tempdates]

            # Sort the datetime objects
            sorted_dates = sorted(date_objects)

            # Calculate the median date
            n = len(sorted_dates)
            if n % 2 == 1:
                median_date = sorted_dates[n // 2]
            else:
                middle1 = sorted_dates[n // 2 - 1]
                middle2 = sorted_dates[n // 2]
                median_date = middle1 + (middle2 - middle1) / 2


            # Optionally, format the mean datetime back to a string
            median_date_string = median_date.strftime('%Y-%m-%d')
            dates.append(median_date_string)
            clusters.append(temp)
            temp = ""
        count += 1

    for i in range(len(clusters)):
        print('curr sample: {}'.format(i))
        # if i==100:
        #     break
        datei = dates[i]
        sentence = clusters[i]
        prompt="""Given the collection of news articles or documents, write a concise summary covering all the important events in documents.
        Summaries:"""+sentence+"""
        Concise Summary: """

        print('***********PROMPT**************')
        print(prompt)

        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(**inputs,max_length=300)
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            print('$$$$$$$$$$$$$$$$$$$^^^^^^^^^^********************')
            # response=response.split('Final Label:')[1]
            print(response)
            print('$$$$$$$$$$$$$$$$$$$^^^^^^^^^^********************')
            print('**************')
            summariesdates.append(datei)
            summaries.append(response)
        except:
            print("Error")
    
    responseListV2=[]
    summariesV2=[]
    finalsummariesV2=[]

    count = 1
    # Strips the newline character
    clusters = []
    dates = []
    dates10 = []
    temp = ""
    tempdates = []
    for line in range(len(summaries)):
        temp = temp + str(summaries[line]) + " "
        tempdates.append(summariesdates[line])
        if count % 20 == 0:
            # Convert date strings to datetime objects
            date_objects = [datetime.strptime(date, "%Y-%m-%d") for date in tempdates]

            # Sort the datetime objects
            sorted_dates = sorted(date_objects)

            # Calculate the median date
            n = len(sorted_dates)
            if n % 2 == 1:
                median_date = sorted_dates[n // 2]
            else:
                middle1 = sorted_dates[n // 2 - 1]
                middle2 = sorted_dates[n // 2]
                median_date = middle1 + (middle2 - middle1) / 2


            # Optionally, format the mean datetime back to a string
            median_date_string = median_date.strftime('%Y-%m-%d')
            dates.append(median_date_string)
            clusters.append(temp)
            temp = ""
        count += 1

    for i in range(len(clusters)):
        print('curr sample: {}'.format(i))
        # if i==100:
        #     break
        datess = dates[i]
        sentence = clusters[i]
        prompt="""Given the collection of news articles or documents, write a concise summary covering all the important events in documents.
        Summaries:"""+sentence+"""
        Concise Summary: """

        print('***********PROMPT**************')
        print(prompt)

        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(**inputs,max_length=300)
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            print('$$$$$$$$$$$$$$$$$$$^^^^^^^^^^********************')
            # response=response.split('Final Label:')[1]
            print(response)
            print('$$$$$$$$$$$$$$$$$$$^^^^^^^^^^********************')
            print('**************')
            dates10.append(datess)
            summariesV2.append(response)
        except:
            print("Error")
        

    filepathV2 = "flan-clusters-summary10" + stories[k] + ".csv"
    f = csv.writer(open(filepathV2, "a+"))

    for i in range(len(summariesV2)):
        f.writerow([dates10[i], summariesV2[i]])

    file_contents = []
    dateset = set()
    t = ""
    for line in range(len(summariesV2)):
        dateset.add(dates10[line])
        
        t = t + " " + str(summariesV2[line])


    # Print or process the file contents as needed
    print(dateset)
    print(t)
    file_contents.append(t)

    opath = datasetpath + stories[k] + "/outputTimelinesTime.csv"
    df = pd.read_csv(opath,header=None)
    df = df.iloc[1:]

    refdate = set()

    for line in range(len(df.values.tolist())):
        data_str = df.values.tolist()[line][0]
        # Remove the outer single quotes and split the string into a list
        data_str = data_str[1:-1]  # Remove the outer single quotes
        date_list = [date.strip() for date in data_str.split(',')]

        # Iterate through the list and print each date
        for date_str in date_list:
            date_str = date_str[1:11]
            refdate.add(date_str)


    ppath  = datasetpath + stories[k] + "/outputTimelines.csv"
    df = pd.read_csv(ppath,header=None)
    resultsBlue = -1
    resultsRogue1 = -1
    resultsRogue2 = -1
    resultsRoguel = -1
    resultsRoguelsum = -1

    for i in range(len(df.values.tolist())):
        resultsBlue = max(resultsBlue, bleu.compute(predictions=file_contents, references=df.values.tolist()[i])['bleu'])
        resultsRogue1 = max(resultsRogue1, rouge.compute(predictions=file_contents, references=df.values.tolist()[i])['rouge1'])
        resultsRogue2 = max(resultsRogue2, rouge.compute(predictions=file_contents, references=df.values.tolist()[i])['rouge2'])
        resultsRoguel = max(resultsRoguel, rouge.compute(predictions=file_contents, references=df.values.tolist()[i])['rougeL'])
        resultsRoguelsum = max(resultsRoguelsum, rouge.compute(predictions=file_contents, references=df.values.tolist()[i])['rougeLsum'])

    print("resultsBlue", resultsBlue)
    print("resultsRogue1", resultsRogue1)
    print("resultsRogue2", resultsRogue2)
    print("resultsRoguel", resultsRoguel)
    print("resultsRoguelsum", resultsRoguelsum)
    dateres = evaluate_dates(dateset, refdate)
    print("Date results: ", dateres)

    blue.append(resultsBlue)
    Rouge1.append(resultsRogue1)
    Rouge2.append(resultsRogue2)
    RougeL.append(resultsRoguel)
    RougeLsum.append(resultsRoguelsum)
    prec.append(dateres['precision'])
    rec.append(dateres['recall'])
    f_score.append(dateres['f_score'])

    filefinal  = "flan-final-result.txt"
    f1 = csv.writer(open(filefinal, "a+"))
    headsen = "\n################# " + stories[k] + " ####################\n"
    f1.writerow([headsen])
    f1.writerow(["resultsBlue: ", resultsBlue])
    f1.writerow(["resultsRogue1: ", resultsRogue1])
    f1.writerow(["resultsRogue2: ", resultsRogue2])
    f1.writerow(["resultsRoguel: ", resultsRoguel])
    f1.writerow(["resultsRoguelsum: ", resultsRoguelsum])
    f1.writerow(["Date results: ", dateres])


    f1.writerow(["Final Blue Score : ", mean(blue)])
    f1.writerow(["Final Rogue1 Score : ", mean(Rouge1)])
    f1.writerow(["Final Rouge2 Score : ", mean(Rouge2)])
    f1.writerow(["Final RogueL Score : ", mean(RougeL)])
    f1.writerow(["Final RougeLsum Score : ", mean(RougeLsum)])
    f1.writerow(["Final precison Score : ", mean(prec)])
    f1.writerow(["Final recall Score : ", mean(rec)])
    f1.writerow(["Final f1 Score : ", mean(f_score)])

    print("Final Blue Score : ", mean(blue))
    print("Final Rogue1 Score : ", mean(Rouge1))
    print("Final Rouge2 Score : ", mean(Rouge2))
    print("Final RogueL Score : ", mean(RougeL))
    print("Final RougeLsum Score : ", mean(RougeLsum))
    print("Final precison Score : ", mean(prec))
    print("Final recall Score : ", mean(rec))
    print("Final f1 Score : ", mean(f_score))
