from functools import partial
from multiprocessing import Pool, Manager

from Runner import Runner
from HelperFunctions import get_now, StartDate


class MultiRunner:
    def __init__(self, study_name='multi_run'):
        self.start_date, self.start_date_for_path = StartDate().get_instance()
        self.device = ['cuda:3', 'cuda:4']
        self.n_jobs = 4
        self.is_parallel = True
        self.semaphore = Semaphore(self.n_jobs)
        self.device_status = {device: 0 for device in self.device}
        self.study_name = study_name

        self.hyper_parameter_set = [
            {'model': 'gru', 'embeddings': ['stanford', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['stanford', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['stanford', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['stanford', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['stanford', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['ntua', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['ntua', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['ntua', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['ntua', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['ntua', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['raw', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['raw', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['raw', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['raw', 'position', 'postag']},
            {'model': 'gru', 'embeddings': ['raw', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['stanford', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['stanford', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['stanford', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['stanford', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['stanford', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['ntua', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['ntua', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['ntua', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['ntua', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['ntua', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['raw', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['raw', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['raw', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['raw', 'position', 'postag']},
            {'model': 'cnn', 'embeddings': ['raw', 'position', 'postag']}
        ]

    def single_run(self, lock=None, hyper_params={}):
        if lock is not None:
            lock.acquire()
        device = self.get_device()
        if lock is not None:
            lock.release()
        runner = Runner(device=device, hyper_params=hyper_params, study_name=self.study_name)
        # runner = TestModel(device=device, hyper_params=hyper_params, study_name=self.study_name)
        runner.run()
        self.release_device(device)

    def multiple_run(self):
        if self.is_parallel:
            with Manager() as manager:
                self.device_status = manager.dict(self.device_status)
                lock = manager.Lock()
                func = partial(self.single_run, lock)
                with Pool(processes=self.n_jobs) as pool:
                    for _ in pool.map(func, [hyper_params for hyper_params in self.hyper_parameter_set]):
                        pass
        else:
            for hyper_params in self.hyper_parameter_set:
                self.single_run(hyper_params=hyper_params)

    def get_device(self):
        sorted_device = sorted(self.device_status.items(), key=lambda x: x[1])
        device = sorted_device[0][0]
        self.device_status[device] += 1
        return device

    def release_device(self, device):
        self.device_status[device] -= 1


class TestRunner:
    def __init__(self, device, hyper_params):
        print('{}: {}'.format(device, hyper_params))

    def run(self):
        import time
        import random
        n = random.randint(2, 10)
        time.sleep(n)


if __name__ == '__main__':
    multirunner = MultiRunner(study_name='200808')
    # multirunner = MultiRunner(study_name='test')
    multirunner.multiple_run()
