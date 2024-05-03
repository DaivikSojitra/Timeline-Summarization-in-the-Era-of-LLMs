import pandas as pd
import csv

# for crisis and T17 give their stories as input
# crisis -> ["egypt","libya", "syria", "yemen"]
# T17 -> ["bpoil", "egypt", "finan", "h1n1", "haiti", "iraq", "libya", "mj", "syria"]
stories = ["Al_Gore", "Angela_Merkel", "Ariel_Sharon", "Arnold_Schwarzenegger", "Bashar_al-Assad", "Bill_Clinton", "Charles_Taylor", "Chris_Brown", "David_Beckham", "David_Bowie",  "Dilma_Rousseff", "Dmitry_Medvedev", "Dominique_Strauss-Kahn", "Edward_Snowden", "Ehud_Olmert", "Enron", "Hamid_Karzai", "Hassan_Rouhani", "Hu_Jintao", "Jacob_Zuma", "John_Boehner", "John_Kerry", "Julian_Assange", "Lance_Armstrong", "Mahmoud_Ahmadinejad", "Marco_Rubio", "Margaret_Thatcher", "Michael_Jackson", "Michelle_Obama", "Mitt_Romney", "Morgan_Tsvangirai", "Nawaz_Sharif", "Nelson_Mandela", "Osama_bin_Laden", "Oscar_Pistorius", "Phil_Spector", "Prince_William", "Robert_Mugabe", "Rupert_Murdoch", "Saddam_Hussein", "Sarah_Palin", "Silvio_Berlusconi", "Steve_Jobs", "Taliban", "Ted_Cruz", "Tiger_Woods", "WikiLeaks"]

for i in range(len(stories)):
    # give time-sent as input which is created by heidel time using news-tls master
    datasetpath = "/Dataset/entities/"+stories[i] + "time-sent.csv"
    df = pd.read_csv(datasetpath,header=None)
    df = df.iloc[1:]
    print(df.head())
    print(df.values.tolist()[0])
    cnt = 0
    dicts = {}
    original = {}
    for line in range(len(df.values.tolist())):
        print(cnt)
        date = df.values.tolist()[line][0]
        text = df.values.tolist()[line][1].split(", '")
        dicts[date] = len(text)
        original[date] = text
        cnt = cnt + 1

    sorted_dict = dict(sorted(dicts.items(), key=lambda item: item[1], reverse=True))

    timeline = {}
    count = 0
    for key, value in sorted_dict.items():
        if count == 150:
            break
        timeline[key] = original[key]
        count = count + 1

    timeline = dict(sorted(timeline.items()))
    # Specify the CSV file name
    csv_file = "timeline-output-"+stories[i]+".csv"

    # Writing the dictionary to a CSV file
    f1 = csv.writer(open(csv_file, "w+"))

    for key,value in timeline.items():
        f1.writerow([key, value])
