from transformers import AutoModelForCausalLM, AutoTokenizer


PATH_TO_LLM = ""
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_LLM)
model = AutoModelForCausalLM.from_pretrained(PATH_TO_LLM, device_map="balanced_low_0")

category = ["Diabetes Mellitus Experimental", "Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2"]

for i in range(len(category)):
    # simple
    prompt = "Suppose you are an expert at medicine and computer science. There are now the following diabetes paper subcategories: Diabetes Mellitus Experimental, Diabetes Mellitus Type 1, and Diabetes Mellitus Type 2. Please write a paper abstract of about 200 words, so that its classification best fits the " + category[i] + " category."
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=300)
    print(category[i] + ":")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
