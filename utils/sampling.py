import numpy as np


def multiscale_timesample(t_start, t_end, dt, sample_separation, sample_duration):
    t = np.arange(t_start, min(t_end, sample_duration)+dt, dt)
    current_end = t[-1]
    while current_end+sample_separation < t_end:
        t = np.hstack((t, np.arange(current_end+sample_separation,
                                         current_end+sample_separation+sample_duration+dt,
                                         dt)))
        current_end = t[-1]
    return t


def multiscale_time_to_subindex(t_start, t_end, dt, sample_separation, sample_duration):
    t = multiscale_timesample(t_start, t_end, dt, sample_separation, sample_duration)
    return ((t-t_start)/dt).astype(int)


def uniform_time_to_index(t):
    return ((t-t[0])/(t[1]-t[0])).astype(int)
