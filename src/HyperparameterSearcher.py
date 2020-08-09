import optuna
from optuna.samplers import TPESampler

from Runner import Runner
from HelperFunctions import get_milliseconds, get_now, StartDate


class HyperparameterSearcher:
    def __init__(self, device='cuda:0', study_name='test'):
        self.direction = 'maximize'
        self.sampler = TPESampler(seed=int(get_milliseconds()))
        self.study_name = study_name
        self.device = device
        # study = optuna.create_study(study_name=study_name, sampler=sampler, direction=direction, storage='sqlite:///./results/hyp-search.db', load_if_exists=True)
        self.study = optuna.create_study(study_name=self.study_name, sampler=self.sampler, direction=self.direction)
        self.start_date, self.start_date_for_path = StartDate().get_instance()

    def get_hyperparameters(self, trial):
        hyper_params = {}
        hyper_params['optimizer_type'] = trial.suggest_categorical('optimizer_type', ['sgd', 'adam'])
        hyper_params['lr'] = trial.suggest_loguniform('lr', 1e-5, 5e-1)
        hyper_params['gradient_clip'] = trial.suggest_uniform('gradient_clip', 0.0, 5.0)
        hyper_params['weight_decay'] = trial.suggest_uniform('weight_decay', 0.0, 1.0)
        hyper_params['dropout_ratio'] = trial.suggest_uniform('dropout_ratio', 0.0, 1.0)
        hyper_params['num_head'] = trial.suggest_int('num_head', 1, 32, 1)
        if hyper_params['optimizer_type'] == 'sgd':
            hyper_params['momentum'] = trial.suggest_uniform('momentum', 0.0, 5.0)
        hyper_params['model'] = trial.suggest_categorical('model', ['gru', 'gru_with_cheating', 'cnn'])
        if hyper_params['model'] == 'cnn':
            hyper_params['kernel_size'] = trial.suggest_int('kernel_size', 50, 300, 5)
            hyper_params['window_size1'] = trial.suggest_int('window_size1', 3, 21, 2)
            hyper_params['window_size2'] = trial.suggest_int('window_size2', 3, 21, 2)
            hyper_params['window_size3'] = trial.suggest_int('window_size3', 3, 21, 2)

        return hyper_params

    def objective(self, trial):
        hyper_params = self.get_hyperparameters(trial=trial)
        runner = Runner(device=self.device, hyper_params=hyper_params, study_name=self.study_name)
        score = runner.run()
        return score

    def run(self):
        self.study.optimize(self.objective,
                            n_trials=100000,
                            catch=(ValueError,),
                            n_jobs=1)
        print(self.study.best_params)


if __name__ == '__main__':
    searcher = HyperparameterSearcher(device='cuda:2', study_name='hyp_search')
    searcher.run()
