import pandas as pd
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM

torch.cuda.set_device(1)
device = torch.device("cuda:1")  # Set the device


tokenizer = LlamaTokenizer.from_pretrained("syzymon/long_llama_3b")
model = AutoModelForCausalLM.from_pretrained("syzymon/long_llama_3b", torch_dtype=torch.float32, 
    mem_layers=[], 
    mem_dtype='bfloat16',
    trust_remote_code=True,
    mem_attention_grouping=(4, 2048))
model = model.to('cuda:1')

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

# for crisis and T17 give their stories as input and chande datasetpath accordingly
# crisis -> ["egypt","libya", "syria", "yemen"]
# T17 -> ["bpoil", "egypt", "finan", "h1n1", "haiti", "iraq", "libya", "mj", "syria"]
stories = ["Al_Gore", "Angela_Merkel", "Ariel_Sharon", "Arnold_Schwarzenegger", "Bashar_al-Assad", "Bill_Clinton", "Charles_Taylor", "Chris_Brown", "David_Beckham", "David_Bowie",  "Dilma_Rousseff", "Dmitry_Medvedev", "Dominique_Strauss-Kahn", "Edward_Snowden", "Ehud_Olmert", "Enron", "Hamid_Karzai", "Hassan_Rouhani", "Hu_Jintao", "Jacob_Zuma", "John_Boehner", "John_Kerry", "Julian_Assange", "Lance_Armstrong", "Mahmoud_Ahmadinejad", "Marco_Rubio", "Margaret_Thatcher", "Michael_Jackson", "Michelle_Obama", "Mitt_Romney", "Morgan_Tsvangirai", "Nawaz_Sharif", "Nelson_Mandela", "Osama_bin_Laden", "Oscar_Pistorius", "Phil_Spector", "Prince_William", "Robert_Mugabe", "Rupert_Murdoch", "Saddam_Hussein", "Sarah_Palin", "Silvio_Berlusconi", "Steve_Jobs", "Taliban", "Ted_Cruz", "Tiger_Woods", "WikiLeaks"]

# for Crisis and T17 in each story provided time-sent.csv take top150 using Top150Creation.py and provide it here
datasetpath = "/Datasset/Entities/top150/timeline-output-"

import csv

for k in range(len(stories)):
    path = datasetpath + stories[k] + ".csv"
    df = pd.read_csv(path,header=None)
    # df = df.iloc[1:]
    print(df.head())
    print(df.values.tolist()[0])

    summariesdates = []
    summaries = []

    for i in range(len(df.values.tolist())):
        print('curr sample: {}'.format(i))
        # if i==100:
        #     break
        datei = df.values.tolist()[i][0]
        sentence = df.values.tolist()[i][1]
        temp = ""
        for j in range(len(sentence)):
            temp += sentence[j]
        prompt="""Given the collection of news articles or documents, write a summary covering all the important events and summary should be consistent, coherent and salient.
        Articles:"""+temp+"""
        Summary: """

        print('***********PROMPT**************')
        print(prompt)

        try:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to('cuda:1')
            generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            num_beams=1,
            last_context_length=1792,
            do_sample=True,
            temperature=1.0,
            )
            # input('enter')
            response = tokenizer.decode(generation_output[0])
            print(response)
            # input('enter')
            response=response.split('Summary:')[1]
            print(response)
            print('$$$$$$$$$$$$$$$$$$$^^^^^^^^^^********************')
            print('**************')
            summariesdates.append(datei)
            summaries.append(response)
        except:
            print("Error")

    
    filepathV2 = "flan-summary" + stories[k] + ".csv"
    f = csv.writer(open(filepathV2, "a+"))

    for i in range(len(summaries)):
        f.writerow([summariesdates[i], summaries[i]])