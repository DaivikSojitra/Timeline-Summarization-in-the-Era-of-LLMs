import pandas as pd
from transformers import pipeline
import torch
# device = torch.device("cpu")
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from datetime import datetime
from evaluate import load
from statistics import mean

bleu = load("bleu")
rouge = load('rouge')

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=4098,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0.7})

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

# for crisis and T17 give their stories as input
# crisis -> ["egypt","libya", "syria", "yemen"]
# T17 -> ["bpoil", "egypt", "finan", "h1n1", "haiti", "iraq", "libya", "mj", "syria"]
stories = ["Al_Gore", "Angela_Merkel", "Ariel_Sharon", "Arnold_Schwarzenegger", "Bashar_al-Assad", "Bill_Clinton", "Charles_Taylor", "Chris_Brown", "David_Beckham", "David_Bowie",  "Dilma_Rousseff", "Dmitry_Medvedev", "Dominique_Strauss-Kahn", "Edward_Snowden", "Ehud_Olmert", "Enron", "Hamid_Karzai", "Hassan_Rouhani", "Hu_Jintao", "Jacob_Zuma", "John_Boehner", "John_Kerry", "Julian_Assange", "Lance_Armstrong", "Mahmoud_Ahmadinejad", "Marco_Rubio", "Margaret_Thatcher", "Michael_Jackson", "Michelle_Obama", "Mitt_Romney", "Morgan_Tsvangirai", "Nawaz_Sharif", "Nelson_Mandela", "Osama_bin_Laden", "Oscar_Pistorius", "Phil_Spector", "Prince_William", "Robert_Mugabe", "Rupert_Murdoch", "Saddam_Hussein", "Sarah_Palin", "Silvio_Berlusconi", "Steve_Jobs", "Taliban", "Ted_Cruz", "Tiger_Woods", "WikiLeaks"]

import csv
import json
from langchain import PromptTemplate,  LLMChain

for k in range(len(stories)):
    path = '/Dataset/Entities/KG/entities-KG-' + stories[k] + '.json'

    with open(path, 'r') as json_file:
        data_list = json.load(json_file)

        path = 'output-' + stories[k] + '.csv'
        f1 = csv.writer(open(path, "a+"))

        # Group data based on date
        grouped_data = {}
        for item in data_list:
            date = item['date']
            if date in grouped_data:
                grouped_data[date].append(item)
            else:
                grouped_data[date] = [item]

        # Print the grouped data
        i = 1
        for date, items in grouped_data.items():
            print(f"Date: {date}")

            print('curr sample: {}'.format(i))
            sentence = str(items)
            template = """
              Given the collection of news articles or documents, write a concise summary covering all the important events in documents.
              Articles: {sentence}
              Summaries:
            """
        
            prompt = PromptTemplate(template=template, input_variables=["sentence"])

            llm_chain = LLMChain(prompt=prompt, llm=llm)

            print('***********PROMPT**************')
            print(prompt)

            try:
                response = llm_chain.run(sentence)
                print(response)
                print('$$$$$$$$$$$$$$$$$$$^^^^^^^^^^********************')
                print('**************')
                f1.writerow([date, response])
                i+=1
            except Exception as e:
                print(f"Error: {e}")