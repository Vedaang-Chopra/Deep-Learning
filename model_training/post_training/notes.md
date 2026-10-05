### Post-Training

Pretraining -> Is bulk of training

Model is trained on particular task, Faster and Cheaper

SFT -> Supervised Finetuning (Train model on Label Prompt-Response pairs) -> Hoping models to follow instructions
New behaviors or changes
DPO -> Direct preference Option -> Show Exact example for good and bad, it moves away from bad (Contrastive Learning)



Online Reinforcement learning -> Using a reward model so that model adapts
GRPO, PPO -> Algorithm 
Human labels act as this, Math problems or Coding Problems






Basic Post-Training Concepts
1. Initally the model has random weights and then we pre-train it on text data
2. Post-Training works on top of base-pretrained model; We get instruct or chat model from this. 
3. Continual Post Training -> Customised models, enhance a behviour, example for coding, or math


