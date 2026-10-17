from transformers import AutoModelForCausalLM, AutoTokenizer

def return_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model