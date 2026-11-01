from transformers import AutoTokenizer


### Alpaca Style Format

class AlpacaFormatOld:
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





class AlpacaFormat:
    @staticmethod
    def to_messages(row):
        user = row["instruction"]
        if row.get("input"):
            user += "\n\nInput:\n" + row["input"]

        return {"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["output"]}
        ]}


class ShareGPTFormat:
    @staticmethod
    def to_messages(row):
        role_map = {
            "human": "user",
            "gpt": "assistant",
            "system": "system",
            "user": "user",
            "assistant": "assistant"
        }
        messages = []
        for msg in row["conversations"]:
            role = role_map[msg["from"]]
            messages.append({
                "role": role,
                "content": msg["value"]
            })
        return {"messages": messages}


class MessagesFormat:
    @staticmethod
    def to_messages(row):
        return {"messages": row["messages"]}


class PromptCompletionFormat:
    @staticmethod
    def to_messages(row):
        prompt = row["prompt"]
        completion = row["completion"]

        if isinstance(prompt, list) and isinstance(completion, list):
            return {"messages": prompt + completion}

        # Valid when prompt means a user instruction
        return {"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]}


class FlanFormat:
    @staticmethod
    def to_messages(row):
        return {"messages": [
            {"role": "user", "content": row["inputs"]},
            {"role": "assistant", "content": row["targets"]}
        ]}


class DollyFormat:
    @staticmethod
    def to_messages(row):
        user = row["instruction"]
        if row.get("context"):
            user += "\n\nContext:\n" + row["context"]

        return {"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["response"]}
        ]}


class SFTDatasetPrepare:
    
    ADAPTERS = {
        "alpaca": AlpacaFormat,
        "sharegpt": ShareGPTFormat,
        "messages": MessagesFormat,
        "flan": FlanFormat,
        "dolly": DollyFormat,
    }

    def __init__(self, model_name = None, schema ="auto", template="auto", prompt_as_instruction=False):
        
        self.schema = schema
        self.template = template
        self.prompt_as_instruction = prompt_as_instruction

        self.tokenizer = (
            AutoTokenizer.from_pretrained(model_name)
            if model_name else None
        )
    
    def detect_schema(self, row):
        if self.schema !="auto":
            return self.schema
        
        keys = set(row.keys())
        if "messages" in keys:
            return "messages"
        if "conversations" in keys:
            return "sharegpt"
        if {"instruction", "context", "response"} <= keys:
            return "dolly"
        if {"instruction", "output"} <= keys:
            return "alpaca"
        if {"inputs", "targets"} <= keys:
            return "flan"
        if {"prompt", "completion"} <= keys:
            return "prompt_completion"
        if {"input", "output"} <= keys:
            return "input_output"
        if "text" in keys:
            return "text"

        raise ValueError(f"Unknown schema: {keys}")
    
    @staticmethod
    def make_pair(user, assistant):
        return {
            "prompt": [
                {"role": "user", "content": user}
            ],
            "completion": [
                {"role": "assistant", "content": assistant}
            ],
        }
    
    def convert(self, row):
        schema = self.detect_schema(row)
        
        if schema =="text":
            return {"text": row["text"]}
        
        if schema == "prompt_completion":
            prompt = row["prompt"]
            completion = row["completion"]

            if (
                self.prompt_as_instruction
                and isinstance(prompt, str)
                and isinstance(completion, str)
            ):
                return self.make_pair(prompt, completion)

            return {
                "prompt": prompt,
                "completion": completion
            }
            
        if schema == "input_output":
            return self.make_pair(
                row["input"], row["output"]
            )
        
        # Use existing formatter classes
        adapter = self.ADAPTERS[schema]
        messages = adapter.to_messages(row)["messages"]

        # Single-turn supervised datasets
        if schema in {"alpaca", "flan", "dolly"}:
            return {
                "prompt": messages[:-1],
                "completion": messages[-1:]
            }

        # Multi-turn datasets
        return {"messages": messages}

    
    
    def prepare(self, dataset):
        if isinstance(dataset, list):
            return [self.convert(row) for row in dataset]

        return dataset.map(
            self.convert,
            remove_columns=dataset.column_names
        )
    
    def preview(self, row):
        # Plain text datasets
        if "text" in row:
            return row["text"]

        if "messages" in row:
            messages = row["messages"]
        else:
            prompt = row["prompt"]
            completion = row["completion"]

            # Raw text continuation: no chat conversion
            if isinstance(prompt, str):
                return prompt + completion

            messages = prompt + completion

        if self.template == "alpaca":
            if (
                len(messages) != 2
                or [m["role"] for m in messages]
                != ["user", "assistant"]
            ):
                raise ValueError(
                    "Alpaca template supports one user/assistant pair"
                )

            return (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes "
                "the request.\n\n"
                f"### Instruction:\n{messages[0]['content']}"
                f"\n\n### Response:\n{messages[1]['content']}"
            )

        if self.template not in {"auto", "model"}:
            raise ValueError("Unsupported template")

        if (
            self.tokenizer is None
            or not self.tokenizer.chat_template
        ):
            raise ValueError(
                "No model chat template available. "
                "Provide a compatible model or use template='alpaca'."
            )

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
