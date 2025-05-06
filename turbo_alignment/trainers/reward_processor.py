from abc import ABC, abstractmethod

import torch

from turbo_alignment.common.registry import Registrable
from turbo_alignment.dist_utils.utils import get_global_mean, get_log_mean_std
from turbo_alignment.settings.pipelines.train.online import RewardProcessorType


class RewardProcessor(ABC, Registrable):
    def __init__(
        self,
        mean_baseline_coef: float,
        num_generations: int,
    ) -> None:
        self.mean_reward: float | None = None
        self.mean_baseline_coef = mean_baseline_coef
        self.num_generations = num_generations

    @abstractmethod
    def postprocess_rewards(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def baseline_rewards(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        baseline: float = self.mean_reward if self.mean_reward is not None else 0
        advantages: torch.Tensor = rewards - baseline

        with torch.no_grad():
            metrics: dict[str, float | int] = {'baseline_mean': baseline}

        self.update_mean_reward(rewards=rewards)

        return advantages, metrics

    @torch.no_grad()
    def update_mean_reward(self, rewards: torch.Tensor):
        global_mean_reward: float = get_global_mean(rewards)

        if self.mean_reward is None:
            self.mean_reward = global_mean_reward
        else:
            self.mean_reward = (
                self.mean_baseline_coef * self.mean_reward + (1 - self.mean_baseline_coef) * global_mean_reward
            )


@RewardProcessor.register(RewardProcessorType.REINFORCE)
class REINFORCERewardProcessor(RewardProcessor):
    def baseline_rewards(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        return rewards, {}

    def postprocess_rewards(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        rewards = rewards[:, 0].unsqueeze(-1)  # values are at 1

        with torch.no_grad():
            metrics: dict[str, float] = get_log_mean_std(rewards, 'real_reward', use_global=True)  # FIXME?

        return rewards, metrics


@RewardProcessor.register(RewardProcessorType.REINFORCE_WITH_BASELINE)
class REINFORCEWithBaselineRewardProcessor(RewardProcessor):
    def postprocess_rewards(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        rewards = rewards[:, 0].unsqueeze(-1)  # values are at 1

        with torch.no_grad():
            metrics: dict[str, float] = get_log_mean_std(rewards, 'real_reward', use_global=True)  # FIXME?

        return rewards, metrics


@RewardProcessor.register(RewardProcessorType.RLOO)
class RLOORewardProcessor(RewardProcessor):
    def postprocess_rewards(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        rewards = rewards[:, 0].unsqueeze(-1)  # values are at 1

        with torch.no_grad():
            metrics: dict[str, float] = get_log_mean_std(rewards, 'real_reward', use_global=True)  # FIXME?

        return rewards, metrics

    def baseline_rewards(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        rewards = rewards.reshape(-1, self.num_generations)
        baseline: torch.Tensor = (rewards.sum(-1).unsqueeze(-1) - rewards) / (self.num_generations - 1)
        rloo_advantages: torch.Tensor = (rewards - baseline).flatten()

        with torch.no_grad():
            metrics: dict[str, float] = {
                **get_log_mean_std(baseline, 'baseline', use_global=True),  # FIXME?
                **get_log_mean_std(baseline.std(-1), 'baseline_inner_std', use_global=True),  # FIXME?
            }

        return rloo_advantages, metrics


@RewardProcessor.register(RewardProcessorType.GRPO)
class GRPORewardProcessor(RewardProcessor):
    def __init__(self, num_generations) -> None:
        super().__init__(num_generations=num_generations, mean_baseline_coef=0.0)  # FIXME

    def postprocess_rewards(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        rewards = rewards[:, 0].unsqueeze(-1)  # values are at

        with torch.no_grad():
            metrics: dict[str, float] = get_log_mean_std(rewards, 'real_reward', use_global=True)  # FIXME?

        return rewards, metrics

    def baseline_rewards(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        rewards = rewards.reshape(-1, self.num_generations)  # [bs, num_generations]

        mean_grouped_rewards = rewards.mean(dim=1)  # [bs,]
        std_grouped_rewards = rewards.std(dim=1)  # [bs,]

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0).reshape(
            -1, self.num_generations
        )  # [bs * num_generations], [bs, num_generations]
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0).reshape(
            -1, self.num_generations
        )  # [bs * num_generations], [bs, num_generations]

        grpo_advantages = ((rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)).flatten()

        with torch.no_grad():
            metrics: dict[str, float] = {
                **get_log_mean_std(mean_grouped_rewards, 'mean_group_advantage', use_global=True),  # FIXME?
                **get_log_mean_std(std_grouped_rewards, 'std_group_advantage', use_global=True),  # FIXME?
            }

        return grpo_advantages, metrics
