import numpy as np

NAMES = ['log_span_ms','log_iat_mean_ms','log_iat_sd_ms','log_iat_median_ms',
         'log_iat_p90_ms','log_iat_p99_ms','log_iat_max_ms','iat_burstiness',
         'outgoing_fraction','switch_fraction','log_run_count','log_run_mean',
         'log_run_sd','log_run_median','log_run_p90','log_run_max']


def light_features(direction, times, expected_length=526):
    direction = np.asarray(direction)
    times = np.asarray(times,dtype=np.float64)
    if direction.ndim!=1 or times.shape!=direction.shape or len(direction)!=expected_length:
        raise ValueError('Direction and timestamp inputs must match the declared length')
    if not np.isin(direction,[-1,1]).all() or not np.isfinite(times).all():
        raise ValueError('Invalid direction or timestamp input')
    gaps = np.maximum(np.diff(times),0.)*1000.
    mean, sd = gaps.mean(), gaps.std(ddof=0)
    boundaries = np.r_[0,np.flatnonzero(direction[1:]!=direction[:-1])+1,len(direction)]
    runs = np.diff(boundaries)
    values = [np.log1p(max(0.,times[-1]-times[0])*1000.),np.log1p(mean),np.log1p(sd),
              np.log1p(np.median(gaps)),np.log1p(np.quantile(gaps,.9)),np.log1p(np.quantile(gaps,.99)),
              np.log1p(gaps.max()),(sd-mean)/(sd+mean) if sd+mean else 0.,
              np.mean(direction==1),np.mean(direction[1:]!=direction[:-1]),np.log1p(len(runs)),
              np.log1p(runs.mean()),np.log1p(runs.std(ddof=0)),np.log1p(np.median(runs)),
              np.log1p(np.quantile(runs,.9)),np.log1p(runs.max())]
    result = np.asarray(values,dtype=np.float32)
    assert result.shape==(16,) and np.isfinite(result).all()
    return result
