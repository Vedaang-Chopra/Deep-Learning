


### Alpaca Style Format

class AlpacaFormat:
    def __init__(self, data) -> None:
        self.data = data
        # self.instruction_text = 
    
    
    def format_input(self, instruction:str, in_text:str):
        instruction_text =(
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{instruction}"
        ) ## Shorter helps in token saving
        
        input_text =  f"\n\n #### Input:\n{in_text}" if in_text else ""
        return instruction_text + input_text

    def detailed_response(self, out_text:str):
        desired_response = f"\n\n### Response:\n{out_text}"
        
        return desired_response
    
    def complete_response(self, instruction, in_text, out_text):
        return (self.format_input(instruction, in_text) + self.detailed_response(out_text))
    