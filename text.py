from transformers import pipeline
import random
import pandas as pd
from datetime import datetime

print("Transformers library succesfully loaded")

print("Generator model loaded")

texts = []
labels = []
  
df = pd.DataFrame(columns=["model","label","text"])

for i in range(10):
  print(f"Start of {i} cycle")
  for model in ["neo27B","GPT2-L"]:
    print(f"Model {model} started to generate texts")
    if model == "neo27B":
      generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B", device=0)
    if model == "GPT2-L":
      generator = pipeline("text-generation", model="gpt2-large", device=0)
    for j in [700, 600, 500, 400, 300, 200]:
      for topic in ["basketball","boxing","gaming"]:
        if topic == "basketball":
          keywords = ["About basketball:", "About basketball ball:", "About basketball player:", "basketball", "basketball player", "basketball game", "basketball history", "basketball court", "basketball fans", "NBA", "Euroleague", "WNBA"]
        if topic == "boxing":
          keywords = ["About boxing:", "About boxing gloves:", "About boxing champion:", "boxing", "boxing champion", "boxing match", "boxing history", "boxing ring", "boxing arena"]
        if topic == "gaming":
          keywords = ["About gaming:", "About gaming setup:", "About gaming champion:", "gaming", "gaming champion", "gaming match", "gaming history", "gaming tournament", "League of Legends", "World of Warcraft", "GTA", "Minecraft", "Red Dead Redemption 2", "Call of Duty", "Super Mario", "Fortnite"]
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

            df = df.append({"model": model,
                            "label": topic,
                            "text": text
                           }, ignore_index=True)           

          except:
            print("Error in generation. Skip")
          try:  
            print(len(df))
          except:
            print("Error in len(texts)")
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%m")
    df.to_csv(f"Texts_{time_now}.csv")
  
df.to_csv("Generated_texts.csv")
