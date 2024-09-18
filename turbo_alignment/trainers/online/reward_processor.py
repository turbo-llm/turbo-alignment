from abc import ABC
import torch

from turbo_alignment.common.distributed import get_global_mean, get_log_mean_std

from allenai_common import Registrable

from turbo_alignment.settings.online import RewardProcessorType


class RewardProcessor(ABC, Registrable):

    def __init__(
        self,
        mean_baseline_coef: float,
    ) -> None:
        self.mean_reward = None
        self.mean_baseline_coef = mean_baseline_coef

    def postprocess_rewards(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def baseline_rewards(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        baseline: float = self.mean_reward if self.mean_reward is not None else 0
        advantages: torch.Tensor = rewards - baseline

        with torch.no_grad():
            metrics: dict[str, float | int] = {"baseline_mean": baseline}

        self.update_mean_reward(rewards=rewards)

        return advantages, metrics
    
    @torch.no_grad()
    def update_mean_reward(self, rewards: torch.Tensor):
        global_mean_reward: float = get_global_mean(rewards)

        if self.mean_reward is None:
            self.mean_reward: float = global_mean_reward
        else:
            self.mean_reward = (
                self.mean_baseline_coef * self.mean_reward
                + (1 - self.mean_baseline_coef) * global_mean_reward
            )


@RewardProcessor.register(RewardProcessorType.RLOO)
class RLOORewardProcessor(RewardProcessor):

    def postprocess_rewards(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        rewards = rewards[:, 0].unsqueeze(-1)  # values are at 1

        with torch.no_grad():
            metrics: dict[str, float] = get_log_mean_std(rewards, "real_reward")

        return rewards, metrics

    def baseline_rewards(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        # FIXME: get sampling_params.n from rewards shape?
        rewards = rewards.reshape(-1, self.sampling_params.n)
        baseline: torch.Tensor = (rewards.sum(-1).unsqueeze(-1) - rewards) / (
            self.sampling_params.n - 1
        )
        rloo_advantages: torch.Tensor = (rewards - baseline).flatten()

        with torch.no_grad():
            metrics: dict[str, float] = {
                **get_log_mean_std(baseline, "baseline"),
                **get_log_mean_std(baseline.std(-1), "baseline_inner_std"),
            }

        return rloo_advantages, metrics
