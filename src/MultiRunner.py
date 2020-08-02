from multiprocessing import Process, Manager
from multiprocessing import Semaphore

from Runner import Runner
from HelperFunctions import get_now, StartDate


class MultiRunner:
    def __init__(self):
        self.start_date, self.start_date_for_path = StartDate().get_instance()
        self.device = ['cuda:3', 'cuda:4']
        self.n_jobs = 4
        self.is_parallel = True
        self.semaphore = Semaphore(self.n_jobs)
        self.device_status = {device: 0 for device in self.device}

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

    def single_run(self, hyper_params={}):
        self.semaphore.acquire()
        device = self.get_device()
        runner = Runner(device=device, hyper_params=hyper_params)
        runner.run()
        self.release_device(device)
        self.semaphore.release()

    def multiple_run(self):
        if self.is_parallel:
            with Manager() as manager:
                self.device_status = manager.dict(self.device_status)
                all_processes = []
                for hyper_params in self.hyper_parameter_set:
                    p = Process(target=self.single_run, args=(hyper_params,))
                    all_processes.append(p)
                    p.start()
                for p in all_processes:
                    p.join()
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
    multirunner = MultiRunner()
    multirunner.multiple_run()
