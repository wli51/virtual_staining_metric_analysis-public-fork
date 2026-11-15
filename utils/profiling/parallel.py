"""Bounded, spawn-based CP FOV workers; parent owns the run lock and manifest."""
import json
import multiprocessing
import os
import signal
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

_STATE = None
_THREAD_LIMIT = None
_THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


@contextmanager
def _worker_environment(threads):
    # Spawned interpreters import numpy before their initializer runs. Set these
    # in the parent for the pool lifetime, then restore the caller's environment.
    previous = {key: os.environ.get(key) for key in _THREAD_VARIABLES}
    try:
        for key in _THREAD_VARIABLES:
            os.environ[key] = str(threads)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _initialize(roots, output, crop_policy, spec, threads, stop_event):
    global _STATE, _THREAD_LIMIT
    # Parent handles Ctrl-C, stops submissions and waits for current shards.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import pyarrow as pa
    from threadpoolctl import threadpool_limits

    from utils.fov_ingestion.collection import MicroscopyCollection

    from .cp_measure import CPMeasureProfiler

    _THREAD_LIMIT = threadpool_limits(limits=threads)
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(1)
    sources = [MicroscopyCollection(root) for root in roots]
    if [s.fingerprint for s in sources] != [s['fingerprint'] for s in spec['sources']]:
        raise ValueError('Source collections changed before worker startup')
    profiler = CPMeasureProfiler(**spec['profiler']['settings'])
    if json.loads(json.dumps(profiler.describe())) != spec['profiler']:
        raise ValueError('Worker CP configuration differs from parent')
    _STATE = (sources[0], sources, Path(output), profiler, crop_policy, spec, stop_event)


def _run(task):
    from concurrent.futures import CancelledError

    from .runner import _profile_fov
    reference, sources, output, profiler, policy, spec, stop_event = _STATE
    if stop_event.is_set():
        raise CancelledError('Profiling stopped before FOV')
    fid, anchors = task
    return _profile_fov(fid, reference, sources, output, profiler, policy, spec,
                        anchors_override=anchors, stop_event=stop_event)


def parallel_fovs(tasks, reference, sources, output, crop_policy, spec, workers, threads):
    """Yield completed FOV records; at most 2*workers tasks are queued/running.

    On failure/Ctrl-C, queued work is cancelled and active workers finish their
    current source shard, then stop. The caller's run lock stays held until all
    processes exit. Successful shard commits survive a failed overall run.
    """
    # Fail before starting workers if the explicit thread-control dependency is absent.
    import threadpoolctl  # noqa: F401

    ctx = multiprocessing.get_context('spawn')
    stop_event = ctx.Event()
    tasks = iter(tasks)
    pending = set()
    with _worker_environment(threads):
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
            initializer=_initialize,
            initargs=([str(s.root) for s in sources], str(output), crop_policy, spec, threads, stop_event))
        try:
            def submit_one():
                try:
                    task = next(tasks)
                except StopIteration:
                    return False
                pending.add(pool.submit(_run, task))
                return True

            for _ in range(2 * workers):
                if not submit_one():
                    break
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                # Observe all failures in this completed set before submitting more.
                results = [future.result() for future in finished]
                for result in results:
                    yield result
                    submit_one()
        finally:
            stop_event.set()
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
