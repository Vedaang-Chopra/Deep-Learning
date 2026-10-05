# POST TRAINING NOTES

THESE ARE NOTES for the RLHF Nathan Lambert -> Youtube playlist


## Lecture-1 -: ML Foundations

1. Basic Deep Learning, PyTorch, Optimization, Probability, RL 
2. LLM Basics



## Lecture-2: - Foundations of RL

Language models apply prob. to text, Transformers are auto-regressive is nature. 

Models predict distribution over time.

Models are Trillions of parameters. Transformer didn't build attention, they scaled it properly. 


Pre-Training currently on trillions of tokens. 

Training Stages: - 
1. Pre-Training - Builds the model's world knowledge, language fluency and broad capabilities
2. Instruction-Tuning/SFT- Teaches the model to answer in a question-answer format and often teaches it to repeat specific token sequences. (If model repeats certain phrase, it comes from this data)
3. Preference-Tuning/RLHF - uses contrastive loss to modify completions as a whole, making the model more richer and flexible 
4. RLVR - Enhanace the model's ability on verifiable questions, which can translate into more complex agentic behaviors 


### TO DO List

1. KL-divergence loss, and Basic-RL
2. I need to look at BERT once. 
3. GPT-2 Scaling laws paper
4. GPT3 - one shot and zero shot 
 