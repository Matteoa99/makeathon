import json

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax


with open('unified.json') as user_file:
    parsed_json = json.load(user_file)



MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

i=0

ris=""
for obj in parsed_json:
    try:
        text = obj["review"]
    # print(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

    # Print labels and scores
        ranking = np.argsort(scores)
        ranking = ranking[::-1]

    #for i in range(scores.shape[0]):
        l = config.id2label[ranking[0]]
        s = scores[ranking[0]]
        obj["sentiment"]=l
        obj["sentiment_accuracy"]= np.round(float(s), 4)
        obj["ranking"] = round((obj["rating"]*np.round(float(s), 4)), 4)
        ris += json.dumps(obj) + "\n"
        #print(f"{text}: {l} {np.round(float(s), 4)}")
    except:
        print("ERROR AT I ", {i})
    i=i+1
    print(i)


file = open("result.json", "w", encoding="utf-8")
file.write(ris)
file.close()
