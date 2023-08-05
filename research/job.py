import numpy as np
import pandas as pd
import datetime as dt
import time 
import copy_reg
import types
import multiprocessing as mp

def _pickle_method(method):
    func_name=method.im_func.__name__
    obj=method.im_self
    cls=method.im_class
    return _unpickle_method, (func_name,obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func=cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType,_pickle_method,_unpickle_method)

def job(r,width=.5):
    # find the index of the earliest barrier touch
    t,p={},np.log((1+r).cumprod(axis=0))
    for j in range(r.shape[1]): # go through columns
        for i in range(r.shape[0]): # go through rows
            if p[i,j]>=width or p[i,j]<=-width:
                t[j]=i
                continue
    return t

def linParts(numAtoms, numThreads):
    # partition of atoms with a single loop
    parts=np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
    parts=np.ceil(parts).astype(int)
    return parts

def mp_pandas_job(func, pd_obj, num_threads=24, num_batches=1, lin_mols=True, **kwargs):
    """
    Parallelize jobs, return a Dataframe or Series
    - func: function to be parallelized. returns a part of the df.
    - pdObj[0]: Name of arg used to pass the molecule
    - pdObj[1]: List of atoms that will be grouped into molecules
    - kwargs: any other args
    """
    # if molecules are linearly splittable
    if linMols:
        parts = linParts(len(pd_obj[1]), num_threads*num_batches)
    else:
        parts = nestedParts(len(pd_obj[1]), num_threads*num_batches)
    jobs = []
    for i in range(1, len(parts)):
        job = {
            pd_obj[0]: pd_obj[1][parts[i-1]:parts[i]],
            'func': func
        }
        job.update(kwargs)  # updates dict with kwargs
        jobs.append(job)
    out = processJobs_(jobs) if num_threads == 1 else processJobs(jobs, num_threads)
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    for i in out:
        df0 = df0.append(i)
    return df0.sort_index()

def processJobs_(jobs):
    out = []
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
    return out

def report_progress(job_num, num_jobs, time_0, task):
    msg = [job_num / num_jobs, (time.time() - time_0, task)]
    msg.append(msg[1] * (1 / msg[0] - 1))
    timestamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = f"{timestamp} {(msg[0] * 100):.2f}% {task} done after {msg[1]:.2f} minutes. Remaining: {msg[2]:.2f} minutes." 
    if job_num < num_jobs:
        print(msg, end='\r')
    else:
        print(msg, end='\n')

def process_jobs(jobs, task=None, numThreads=24):
    if task is None:
        task = jobs[0]['func'].__name__
    with mp.pool(processes=numThreads) as pool:
        outputs, out, time_0 = pool.imap_unordered(expandCall, jobs), [], time.time()
        for i, out_ in enumerate(outputs, 1):
            out.append(out_)
            report_process(i, len(jobs), time_0, task)
    return out
    
def expandCall(kwargs):
    func = kwargs['func']
    del kwargs['func']
    out = func(**kwargs)
    return out
































