from transformers import AutoModelForCausalLM, AutoTokenizer


PATH_TO_LLM = ""
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_LLM)
model = AutoModelForCausalLM.from_pretrained(PATH_TO_LLM, device_map="balanced_low_0")

category = ["Computational linguistics","Databases", "Operating systems", "Computer architecture", "Computer security", "Internet protocols",
                "Computer file systems", "Distributed computing architecture", "Web technology", "Programming language topics"]

for i in range(len(category)):
    # simple
    prompt = "Suppose you are an expert at computer science. There are now the following entity subcategories: Computational linguistics, Databases, Operating systems, Computer architecture, Computer security, Internet protocols, Computer file systems, Distributed computing architecture, Web technology, and Programming language topics. Please write a entity description of about 200 words, so that its classification best fits the " + category[i] + " category."
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=300)
    print(category[i] + ":")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
