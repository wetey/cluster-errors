'''
    script to generate the clustering task portion of the survey
'''
#load necessary packages
import sys
import json

dataset_path = sys.argv[1] #JSON file generate from running clustering_questions.py
save_to = sys.argv[2] #path to save the txt file

#read the JSON file
with open(dataset_path, encoding = 'utf-8') as file:
    data = json.load(file)

prompt = "The following are 5 examples from an offensive language dataset. Only 4 out of the 5 examples were put into the same group. Which example, do you think, belongs to a different group?"

#format the question how Qualtrics accepts
survey = "[[AdvancedFormat]]\n"
for examples in data:
    if len(examples['choices']) == 5:
        survey += "[[Question:MC:SingleAnswer]]\n"
        survey +=   prompt + "\n"
        survey += "[[Choices]]\n"
        current = 1
        for choice in examples['choices']:
            survey += str(current) + ". " + choice + "\n"
            current += 1
        survey += "\n\n"
    

with open(save_to, 'w', encoding='utf8') as file:
    file.write(survey)