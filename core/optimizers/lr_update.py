from math import cos, pi
import numpy as np

class LrUpdater(object):
    """学习率更新器基类，用于回归任务的学习率调整。

    Args:
        by_epoch (bool): 学习率按周期变化。
        warmup (str): 热身类型，可以是 'constant'、'linear' 或 'exp'。
        warmup_iters (int): 热身阶段的迭代次数。
        warmup_ratio (float): 热身开始时学习率与初始学习率的比例。
        warmup_by_epoch (bool): 是否按周期进行热身。
    """

    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 warmup_by_epoch=False):
        # 验证热身参数
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(f'"{warmup}" 不是支持的热身类型，可用类型包括 "constant"、"linear" 和 "exp"')
            assert warmup_iters > 0, '"warmup_iters" 必须是正整数'
            assert 0 < warmup_ratio <= 1.0, '"warmup_ratio" 必须在 (0, 1] 范围内'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # 所有参数组的初始学习率
        self.regular_lr = []  # 如果没有进行热身预计的学习率

    def _set_lr(self, runner, lr_groups):
        """设置优化器的新学习率"""
        for param_group, lr in zip(runner.get("optimizer").param_groups, lr_groups):
            param_group['lr'] = lr

    def get_lr(self, runner, base_lr):
        """获取更新后的学习率"""
        raise NotImplementedError

    def get_regular_lr(self, runner):
        """获取常规学习率"""
        return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        """获取热身阶段的学习率"""

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio ** (1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_run(self, runner):
        """在训练开始前记录初始学习率"""
        for group in runner.get("optimizer").param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [group['initial_lr'] for group in runner.get("optimizer").param_groups]

    def before_train_epoch(self, runner):
        """在每个训练周期之前更新学习率"""
        if self.warmup_iters is None:
            epoch_len = len(runner.get("train_loader"))
            self.warmup_iters = self.warmup_epochs * epoch_len
        if not self.by_epoch:
            return

        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner):
        """在每次训练迭代之前更新学习率"""
        cur_iter = runner.get("iter")
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)


class StepLrUpdater(LrUpdater):
    """分步学习率调度器，适用于回归任务。

    Args:
        step (int | list[int]): 学习率衰减的步长。如果是整数，则表示衰减间隔。如果是列表，则在这些步骤衰减学习率。
        gamma (float, optional): 学习率衰减系数。默认为 0.1。
        min_lr (float, optional): 保持的最小学习率。如果衰减后的学习率低于 `min_lr`，则会将其剪裁到这个值。默认值为 None。
    """

    def __init__(self, step, gamma=0.1, min_lr=None, **kwargs):
        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        super(StepLrUpdater, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.get('epoch') if self.by_epoch else runner.get('iter')

        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break

        lr = base_lr * (self.gamma**exp)
        if self.min_lr is not None:
            lr = max(lr, self.min_lr)
        return lr


class PolyLrUpdater(LrUpdater):

    def __init__(self, power=1., min_lr=0., **kwargs):
        self.power = power
        self.min_lr = min_lr
        super(PolyLrUpdater, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner['epoch']
            max_progress = runner['max_epochs']
        else:
            progress = runner['iter']
            max_progress = runner['max_iters']
        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


class CosineAnnealingLrUpdater(LrUpdater):
    """余弦退火学习率调度器，适用于回归任务。

    Args:
        min_lr (float, optional): 退火后的最小学习率。默认为 None。
        min_lr_ratio (float, optional): 退火后的最小学习率比例。默认为 None。
    """

    def __init__(self, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super(CosineAnnealingLrUpdater, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.get('epoch')
            max_progress = runner.get('max_epochs')
        else:
            progress = runner.get('iter')
            max_progress = runner.get('max_iters')

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr
        return annealing_cos(base_lr, target_lr, progress / max_progress)

class CosineAnnealingCooldownLrUpdater(LrUpdater):
    """带冷却的余弦退火学习率调度器，适用于回归任务。

    Args:
        min_lr (float, optional): 退火后的最小学习率。默认值为 None。
        min_lr_ratio (float, optional): 退火后的最小学习率比例。默认值为 None。
        cool_down_ratio (float): 冷却比率。默认值为 0.1。
        cool_down_time (int): 冷却时间。默认值为 10。
    """

    def __init__(self,
                 min_lr=None,
                 min_lr_ratio=None,
                 cool_down_ratio=0.1,
                 cool_down_time=10,
                 **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.cool_down_time = cool_down_time
        self.cool_down_ratio = cool_down_ratio
        super(CosineAnnealingCooldownLrUpdater, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.get('epoch')
            max_progress = runner.get('max_epochs')
        else:
            progress = runner.get('iter')
            max_progress = runner.get('max_iters')

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        if progress > max_progress - self.cool_down_time:
            return target_lr * self.cool_down_ratio
        else:
            max_progress = max_progress - self.cool_down_time

        return annealing_cos(base_lr, target_lr, progress / max_progress)

def annealing_cos(start, end, factor, weight=1):
    """计算余弦退火学习率。

    Args:
        start (float): 退火的起始学习率。
        end (float): 退火的结束学习率。
        factor (float): 计算当前百分比的系数。范围为 0.0 到 1.0。
        weight (float, optional): 计算实际起始学习率时的组合因子。默认值为 1。
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out



class ReduceLROnPlateauLrUpdater(LrUpdater):
    """Reduce learning rate when a metric has stopped improving.

    Args:
        mode (str): One of `min`, `max`. In `min` mode, lr will be reduced when
            the quantity monitored has stopped decreasing; in `max` mode it will be
            reduced when the quantity monitored has stopped increasing.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        threshold (float): Threshold for measuring the new optimum, to only focus on significant changes.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode, dynamic_threshold = best * (1 + threshold) in `max`
            mode or best * (1 - threshold) in `min` mode. In `abs` mode, dynamic_threshold = best + threshold in `max`
            mode or best - threshold in `min` mode.
        cooldown (int): Number of epochs to wait before resuming normal operation after lr has been reduced.
        min_lr (float or list): A scalar or a list of scalars. A lower bound on the learning rate of all param groups
            or each group respectively.
    """

    def __init__(self, mode='min', factor=0.1, patience=10, threshold=1e-4,
                 threshold_mode='rel', cooldown=0, min_lr=0, **kwargs):
        super(ReduceLROnPlateauLrUpdater, self).__init__(**kwargs)

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.mode = mode
        self.best = None
        self.num_bad_epochs = 0

        if mode not in ['min', 'max']:
            raise ValueError('Mode ' + mode + ' is unknown!')
        if threshold_mode not in ['rel', 'abs']:
            raise ValueError('Threshold mode ' + threshold_mode + ' is unknown!')

        self.mode_worse = None  # The worse value for the chosen mode.
        self.is_better = None  # The function to compare metrics.
        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode == 'min':
            self.mode_worse = np.inf
            if threshold_mode == 'rel':
                self.is_better = lambda a, best: a < best * (1 - threshold)
            else:
                self.is_better = lambda a, best: a < best - threshold

        if mode == 'max':
            self.mode_worse = -np.inf
            if threshold_mode == 'rel':
                self.is_better = lambda a, best: a > best * (1 + threshold)
            else:
                self.is_better = lambda a, best: a > best + threshold

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def get_lr(self, runner, base_lr):
        """获取调整后的学习率"""
        return [group['lr'] for group in runner['optimizer'].param_groups]

    def step(self, metrics, runner):
        """更新学习率"""
        current = float(metrics)
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # Ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(runner)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, runner):
        for i, param_group in enumerate(runner['optimizer'].param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > 1e-8:
                param_group['lr'] = new_lr
                print(f'ReduceLROnPlateau reducing learning rate from {old_lr:.4e} to {new_lr:.4e}')

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0
