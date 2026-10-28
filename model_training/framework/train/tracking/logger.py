import wandb
from torch.utils.tensorboard import SummaryWriter

class ExperimentLogger:
    def __init__(self, project, config=None, log_dir="runs"):
        self.tb = SummaryWriter(log_dir)
        self.wandb_run = wandb.init(
            project=project,
            config=config
        )

    def log(self, metrics, step):
        wandb.log(metrics, step=step)

        for name, value in metrics.items():
            self.tb.add_scalar(name, value, step)

    def finish(self):
        self.tb.close()
        self.wandb_run.finish()
        
        

# logger = ExperimentLogger(
#     project="gpt2-training",
#     config={"lr": 3e-4, "batch_size": 32}
# )

# for step, batch in enumerate(train_loader):
#     optimizer.zero_grad()

#     loss = model(batch)
#     loss.backward()
#     optimizer.step()

#     logger.log({
#         "train/loss": loss.item(),
#         "train/lr": optimizer.param_groups[0]["lr"]
#     }, step)

# logger.finish()
