import torch
from torchmetrics import Metric


class SearchAreaMetric(Metric):
    """
    Calculates the average search area of predicted spherical regions.

    This metric computes the mean volume of spherical search regions predicted by 
    TrackNET for either t1 or t2 time steps. For t2 predictions, the last position 
    is excluded since there is no next hit to predict.

    The search area is calculated as 4/3 * pi * r^3 for each prediction, where r is 
    the predicted sphere radius. Only valid positions according to the target mask 
    are included in the average.

    Args:
        time_step (str): Which predictions to evaluate - either 't1' or 't2'.
            't1' evaluates predictions for immediate next hits
            't2' evaluates predictions for hits after next

    Example:
        >>> metric = SearchAreaMetric('t1')
        >>> preds = {
        ...     'radius_t1': torch.tensor([[[1.0]], [[2.0]]]),  # batch_size=2, seq_len=1
        ...     'radius_t2': torch.tensor([[[1.5]], [[2.5]]])
        ... }
        >>> target_mask = torch.tensor([[True], [True]])  # batch_size=2, seq_len=1
        >>> metric.update(preds, target_mask)
        >>> metric.compute()
        tensor(18.8496)  # mean volume of spheres with r=1.0 and r=2.0

    Note:
        The metric automatically handles distributed training by summing total_area
        and total_count across all processes before computing the final average.
    """

    def __init__(self, time_step='t1'):
        super().__init__()
        if time_step not in ['t1', 't2']:
            raise ValueError("time_step must be 't1' or 't2'")
        self.time_step = time_step
        self.add_state(
            "total_area",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total_count",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )

    def update(self, preds: dict, target_mask: torch.Tensor):
        radius = preds[f'radius_{self.time_step}']

        # For t2, exclude last prediction as there's no next hit
        if self.time_step == 't2':
            radius = radius[:, :-1]

        # Calculate areas of spheres (4/3 * pi * r^3)
        areas = (4/3) * torch.pi * torch.pow(radius[..., 0], 3)

        # Apply mask for valid predictions
        if self.time_step == 't2':
            mask = target_mask[:, -radius.size(1):]
        else:
            mask = target_mask[:, :radius.size(1)]

        valid_areas = areas[mask]

        self.total_area += valid_areas.sum()
        self.total_count += valid_areas.numel()

    def compute(self):
        return self.total_area / self.total_count


class HitEfficiencyMetric(Metric):
    """
    Calculates the fraction of true hits that lie inside predicted search regions.

    For each prediction, checks if the actual hit position falls within the
    predicted spherical search region (i.e. if the distance between predicted and
    actual position is less than or equal to the predicted radius). The metric
    returns the fraction of hits that satisfy this criterion.

    Args:
        time_step (str): Which predictions to evaluate - either 't1' or 't2'.
            't1' evaluates predictions for immediate next hits
            't2' evaluates predictions for hits after next

    Example:
        >>> metric = HitEfficiencyMetric('t1')
        >>> preds = {
        ...     'coords_t1': torch.tensor([[[1.0, 0.0, 0.0]]]),  # predicted position
        ...     'radius_t1': torch.tensor([[[1.5]]]),  # search radius
        ... }
        >>> targets = torch.tensor([[[2.0, 0.0, 0.0]]])  # true position
        >>> target_mask = torch.tensor([[True]])
        >>> metric.update(preds, targets, target_mask)
        >>> metric.compute()
        tensor(1.0)  # hit is within radius 1.5 of prediction

    Note:
        - For t2 predictions, compares with hits at t+2 and excludes the last position
        - The metric automatically handles distributed training by summing hits_inside
          and total_hits across all processes
    """

    def __init__(self, time_step='t1'):
        super().__init__()
        if time_step not in ['t1', 't2']:
            raise ValueError("time_step must be 't1' or 't2'")
        self.time_step = time_step
        self.add_state(
            "hits_inside",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total_hits",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )

    def update(self, preds: dict, targets: torch.Tensor, target_mask: torch.Tensor):
        coords = preds[f'coords_{self.time_step}']
        radius = preds[f'radius_{self.time_step}']

        # For t2, we compare with next hits and exclude last prediction
        if self.time_step == 't2':
            coords = coords[:, :-1]
            radius = radius[:, :-1]
            targets = targets[:, 1:]

        # Calculate distances between predicted and true positions
        distances = torch.norm(coords - targets, dim=-1)

        # Check if hits are inside spheres (distance <= radius)
        hits_inside = distances <= radius[..., 0]

        # Apply mask for valid predictions
        if self.time_step == 't2':
            mask = target_mask[:, -coords.size(1):]
        else:
            mask = target_mask[:, :coords.size(1)]

        valid_hits = hits_inside[mask]

        self.hits_inside += valid_hits.sum()
        self.total_hits += valid_hits.numel()

    def compute(self):
        return self.hits_inside.float() / self.total_hits
