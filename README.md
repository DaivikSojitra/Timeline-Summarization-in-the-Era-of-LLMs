# Timeline-Summarization-in-the-Era-of-LLMs

## Dataset
for each approach their used dataset can be found in respective folders. For benchmarking dataset can be found [here](https://drive.google.com/drive/folders/1gDAF5QZyCWnF_hYKbxIzOyjT6MSkbQXu?usp=sharing).

- Timeline17
- Crisis
- Entities

## Evaluation
For evaluation of the experiments tilse framework is used which will calculate the Concat-F1, Agree-F1, Align-F1 and Date-F1. please refer [this](https://github.com/smartschat/tilse.git).

Steps to evaluate:
- Get and Install tilse framework in your conda environment.
- Once we have outputTimeline.csv first run tlsinp.py from Evaluation folder to make output that can be taken by tilse framework as input.
- run tilsescript.py from Evaluation folder giving Gold and Predicted timelines as input to the script.