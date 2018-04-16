import atexit
from queue import Empty
from multiprocessing import Process, Queue
from rllab.sampler.utils import rollout
import numpy as np
import time
import tensorflow as tf
import pickle

__all__ = [
    'init_worker',
    'init_plot',
    'update_plot'
]

process = None
queue = None


def _worker_start():
    env = None
    policy = None
    max_length = None
    try:
        newsess = tf.Session()
        newsess.__enter__()
        tf.initialize_all_variables().run()
        while True:
            msgs = {}
            # Only fetch the last message of each type
            while True:
                try:
                    #time.sleep(0.05)
                    msg = queue.get_nowait()
                    msgs[msg[0]] = msg[1:]
                except Empty:
                    break
            if 'stop' in msgs:
                break
            elif 'update' in msgs:
                env, policy_pkl = msgs['update']
                policy = pickle.loads(policy_pkl)
                # env.start_viewer()
            elif 'demo' in msgs:
                param_values, max_length = msgs['demo']
                policy.set_param_values(param_values)
                rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
            else:
                if max_length:
                    # This means keep plot new demos, after the update_plot being called.
                    rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
    except KeyboardInterrupt:
        pass


def _shutdown_worker():
    if process:
        queue.put(['stop'])
        queue.close()
        process.join()


def init_worker():
    global process, queue
    queue = Queue()
    process = Process(target=_worker_start)
    process.start()
    atexit.register(_shutdown_worker)


def init_plot(env, policy):
    queue.put(['update', env, pickle.dumps(policy)])


def update_plot(policy, max_length=np.inf):
    queue.put(['demo', policy.get_param_values(), max_length])
