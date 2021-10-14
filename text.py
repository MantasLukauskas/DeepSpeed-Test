from transformers import pipeline
import random
import pandas as pd

print("Transformers library succesfully loaded")

generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B", device=0)

print("Generator model loaded")

texts = []
labels = []
  
df = pd.DataFrame(columns=["label","text"])

for i in range(1):
  print(f"Start of {i} cycle")
  for j in [200]:
    for topic in ["basketball","boxing"]:
      if topic == "basketball":
        keywords = ["About basketball:", "About basketball ball:", "About basketball player:", "basketball", "basketball player", "basketball game", "basketball history"]
      if topic == "boxing":
        keywords = ["About boxing:", "About boxing gloves:", "About boxing champion:", "boxing", "boxing champion", "boxing match", "boxing history"]
      print("Keywords loaded")
      for start in keywords:
        print(start)
        try:
          
          text = generator(f"About {start}:",
                                 do_sample=True,
                                 max_length=j,
                                 num_return_sequences=2,
                                 temperature=random.uniform(0.95, 1.15),
                                 top_k=random.randint(100, 1000),
                                 top_p=random.uniform(0.95, 1.0),
                                 repetition_penalty=1.0)
          
          
          df = df.append({"label": topic,
                          "text": text[0]["generated_text"]
                         }, ignore_index=True)  
          df = df.append({"label": topic,
                          "text": text[1]["generated_text"]
                         }, ignore_index=True)            

        except:
          print("Error in generation. Skip")
        try:  
          print(len(df))
        except:
          print("Error in len(texts)")

df.to_csv("Generated_texts.csv")
