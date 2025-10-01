import numpy as np
from datetime import datetime
from astropy.time import Time

def log_time_delta(delta):
    return np.log1p(delta)


def sin_encoding(period, values):
    return np.sin(2 * np.pi * np.array(values) / period)


def cos_encoding(period, values):
    return np.cos(2 * np.pi * np.array(values) / period)


def day_ind_interval(date_ref, date):
    return date - date_ref


def encode_time_info(time_stamps, ref_date="190101_000000"):
    """
    Returns a vector with temporal encoding, and days elapsed since a reference date:
    - Log(delta_t) since last acquisition
    - Multiscale Sin/Cos (hour-day, day-month, day-year)
    - Julian date
    - Log(t - t0) since first acquisition
    - Number of days since ref_date (in Julian days)

    time_stamps: astropy.time.Time(1d array of dates)
    ref_date: str with format yymmdd_HHMMSS
    """
    ref_date = Time.strptime(ref_date, '%y%m%d_%H%M%S') # ou self.dates.min(axis=0)
    t = (time_stamps-ref_date).to_value("jd") #Julian Date
    date0 = time_stamps[time_stamps.argmin(axis=0)] #get ref date = first acq date
    argsort_id = time_stamps.argsort(axis=0)
    time_stamps = time_stamps[argsort_id]
    revert_id = np.argsort(argsort_id)
    time_deltas = time_stamps - date0 #get time delta to ref date
    time_deltas = time_deltas.sec #convert to seconds
    #delta t0 = time deltas
    jj0 = date0.jd
    delta_ti = []
    delta_t0 = time_deltas
    jj = []
    hour = []
    day_month = []
    day_year = []
    test_date = []

    for i, date in enumerate(time_stamps):
        if i==0:    
            dt = (date - date0)
            dt = dt.sec
        else:
            dt = (date - time_stamps[i-1])
            dt = dt.sec
        test_date.append(date)
        delta_ti.append(dt)
        jj.append(date.jd - jj0)
        yday = date.yday
        hour.append(date.ymdhms[3] + date.ymdhms[4]/60)
        day_month.append(date.ymdhms[2])
        day_year.append(int(yday.split(":")[1]))

    delta_ti=np.array((delta_ti), dtype=np.float32)
    log_dt0 = np.log1p(delta_t0)
    log_dti = np.log1p(delta_ti)
    
    sin_h = sin_encoding(24, hour)
    cos_h = cos_encoding(24, hour)
    
    sin_m = sin_encoding(30.44, day_month)
    cos_m = cos_encoding(30.44, day_month)
    
    sin_y = sin_encoding(365.25, day_year)
    cos_y = cos_encoding(365.25, day_year)

    sin_jd = sin_encoding(365.25, jj)
    cos_jd = cos_encoding(365.25, jj)
    return_array = np.array(([sin_h, cos_h, sin_m, cos_m, sin_y, cos_y, sin_jd, cos_jd, log_dt0, log_dti]), dtype=np.float32).T
    return return_array[revert_id,:], t