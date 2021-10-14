from transformers import pipeline

print("Transformers library succesfully loaded")

generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B", device=0)

print("Generator model loaded")

texts = []

for i in range(1,5):
  for j in [300,400,500,600]:
    for topic in ["basketball","boxing"]:
      try:
        texts.append(generator(f"About {topic}:", do_sample=True, max_length=j, num_return_sequences=10))
      except:
        print("Error in generation. Skip")
      try:  
        print(len(texts))
      except:
        print("Error in len(texts)")
