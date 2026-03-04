import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def calculate_mse(pred, target):
    """Calculate mean squared error (MSE).

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, 1).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        float: MSE value
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
        f'but got {type(pred)} and {type(target)}.')
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    mse_value = mean_squared_error(target, pred)

    return float(mse_value)

def calculate_rmse(pred, target):
    """Calculate root mean squared error (RMSE).

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, 1).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        float: RMSE value
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
        f'but got {type(pred)} and {type(target)}.')
    mse_value = calculate_mse(pred,target)
    rmse_value = np.sqrt(mse_value)
    return float(rmse_value)

def calculate_r2_score(pred, target):
    """Calculate R^2 (coefficient of determination) regression score function.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, 1).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        float: R^2 score
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
        f'but got {type(pred)} and {type(target)}.')
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    r2_value = r2_score(target, pred)
    return float(r2_value)



def evaluate(
        results,
        gt_labels,
        metric='mse',
        metric_options=None,
        indices=None,
        logger=None):
    """Evaluate the regression dataset.

    Args:
        results (list): Testing results of the dataset.
        metric (str | list[str]): Metrics to be evaluated.
            Default value is `mse`.
        metric_options (dict, optional): Options for calculating metrics.
            Currently not used in this example.
        indices (list, optional): The indices of samples corresponding to
            the results. Defaults to None.
        logger (logging.Logger | str, optional): Logger used for printing
            related information during evaluation. Defaults to None.

    Returns:
        dict: evaluation results
    """
    if metric_options is None:
        metric_options = {}
    if isinstance(metric, str):
        metrics = [metric]
    else:
        metrics = metric
    allowed_metrics = ['mse', 'mae', 'rmse', 'r2','mape','medare']
    eval_results = {}
    if indices is not None:
        gt_labels = gt_labels[indices]
    num_imgs = len(results)
    assert len(gt_labels) == num_imgs, 'Dataset testing results should ' \
                                       'be of the same length as gt_labels.'
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'Metric {invalid_metrics} is not supported.')

    for metric in metrics:
        if metric == 'rmse':
           rmse_value = calculate_rmse(results, gt_labels)
           eval_results['rmse'] = rmse_value
        if metric == 'r2':
           r2_value = calculate_r2_score(results, gt_labels)
           eval_results['r2'] = r2_value

    return eval_results
