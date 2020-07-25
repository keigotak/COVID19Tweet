import os

import torch

from DataPooler import DataPooler
from ValueWatcher import ValueWatcher
from Metrics import get_metrics, get_print_keys
from HelperFunctions import get_now, get_results_path, get_hyperparameter_keys, set_seed, get_save_model_path

from ModelFactory import ModelFactory


class Runner:
    def __init__(self, device='cuda:0', hyper_params={}):
        set_seed()
        self.device = device
        factory = ModelFactory(device=self.device, hyper_params=hyper_params)
        factory_items = factory.generate()
        self.model, self.batchers, self.optimizer, self.criterion = [factory_items[key] for key in ['model', 'batchers', 'optimizer', 'criterion']]
        self.TRAIN_MODE, self.VALID_MODE = 'train', 'valid'
        self.batchers = {self.TRAIN_MODE: self.batchers[self.TRAIN_MODE], self.VALID_MODE: self.batchers[self.VALID_MODE]}
        self.poolers = {self.VALID_MODE: DataPooler()}
        self.valuewatcher = ValueWatcher()
        self.epochs = 100
        self.best_results = {}
        self.hyper_params = factory.hyper_params

    def run(self):
        best_score = 0.0
        save_model_now = get_now(with_path=True)
        for e in range(self.epochs):
            running_loss = {key: 0.0 for key in [self.TRAIN_MODE, self.VALID_MODE]}

            mode = self.TRAIN_MODE
            self.model.train()
            while not self.batchers[mode].is_batch_end():
                self.optimizer.zero_grad()
                x_batch, y_batch = self.batchers[mode].get_batch()
                outputs = self.model(x_batch)
                labels = torch.Tensor(y_batch).float().unsqueeze(1).to(self.device)
                loss = self.criterion(outputs, labels)
                loss.backward()
                running_loss[mode] += loss.item()
                self.optimizer.step()
            self.batchers[mode].reset(with_shuffle=True)

            mode = self.VALID_MODE
            self.model.eval()
            with torch.no_grad():
                while not self.batchers[mode].is_batch_end():
                    x_batch, y_batch = self.batchers[mode].get_batch()
                    outputs = self.model(x_batch)
                    labels = torch.Tensor(y_batch).float().unsqueeze(1).to(self.device)
                    loss = self.criterion(outputs, labels)
                    running_loss[mode] += loss.item()

                    preds = torch.sigmoid(outputs)
                    predicted_label = [0 if item < 0.5 else 1 for item in preds.squeeze().tolist()]
                    self.poolers[mode].set('epoch{}-x'.format(e + 1), x_batch)
                    self.poolers[mode].set('epoch{}-y'.format(e + 1), y_batch)
                    self.poolers[mode].set('epoch{}-logits'.format(e + 1), outputs.squeeze().tolist())
                    self.poolers[mode].set('epoch{}-preds'.format(e + 1), preds.squeeze().tolist())
                    self.poolers[mode].set('epoch{}-predicted_label'.format(e + 1), predicted_label)

            metrics = get_metrics(self.poolers[mode].get('epoch{}-predicted_label'.format(e + 1)),
                                  self.poolers[mode].get('epoch{}-y'.format(e + 1)))
            self.poolers[mode].set('epoch{}-metrics'.format(e + 1), metrics)
            self.poolers[mode].set('epoch{}-train_loss'.format(e + 1), running_loss[self.TRAIN_MODE])
            self.poolers[mode].set('epoch{}-valid_loss'.format(e + 1), running_loss[self.VALID_MODE])
            self.valuewatcher.update(metrics['f1'])
            self.batchers[mode].reset(with_shuffle=False)

            now = get_now()
            text_line = '|'.join(['{} {}: {:.3f}'.format(mode, key, 100 * metrics[key])
                                  if key not in {'tp', 'fp', 'fn', 'tn'}
                                  else '{} {}: {}'.format(mode, key, metrics[key])
                                  for key in get_print_keys()])
            print('{}|epoch: {:3d}|train loss: {:.2f}|valid loss: {:.2f}|{}'.format(now,
                                                                                    e + 1,
                                                                                    running_loss[self.TRAIN_MODE],
                                                                                    running_loss[self.VALID_MODE],
                                                                                    text_line))

            if self.valuewatcher.is_over():
                break
            if self.valuewatcher.is_updated() or e == 0:
                self.best_results = metrics
                self.best_results['date'] = now
                self.best_results['epoch'] = e + 1
                self.best_results['train_loss'] = running_loss[self.TRAIN_MODE]
                self.best_results['valid_loss'] = running_loss[self.VALID_MODE]
                best_score = self.best_results['f1']
                torch.save(self.model.state_dict(),
                           get_save_model_path(dir_tag=save_model_now, file_tag='-{:03}-{:.3f}'.format(e + 1, best_score)))

        self.export_results()

        return best_score

    def export_results(self):
        path = get_results_path('hyp')
        with path.open('a', encoding='utf-8-sig') as f:
            hyper_params = '|'.join(['{}:{}'.format(key, self.hyper_params[key]) for key in get_hyperparameter_keys()])
            f.write(','.join([os.path.basename(__file__)] + [str(self.best_results[key]) for key in ['date', 'epoch', 'train_loss', 'valid_loss'] + get_print_keys()] + [hyper_params]))
            f.write('\n')


if __name__ == '__main__':
    runner = Runner(device='cuda:0')
    score = runner.run()
