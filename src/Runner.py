import os
import time

import torch

from DataPooler import DataPooler
from ValueWatcher import ValueWatcher
from Metrics import get_metrics, get_print_keys
from HelperFunctions import get_now, get_results_path, get_details_path, get_save_model_path, StartDate

from ModelFactory import ModelFactory


class Runner:
    def __init__(self, device='cuda:0', hyper_params={}, study_name=''):
        self.device = device
        factory = ModelFactory(device=self.device, hyper_params=hyper_params)
        factory_items = factory.generate()
        self.model, self.batchers, self.optimizer, self.criterion = [factory_items[key] for key in
                                                                     ['model', 'batchers', 'optimizer', 'criterion']]
        self.TRAIN_MODE, self.VALID_MODE = 'train', 'valid'
        self.batchers = {self.TRAIN_MODE: self.batchers[self.TRAIN_MODE],
                         self.VALID_MODE: self.batchers[self.VALID_MODE]}
        self.poolers = {self.TRAIN_MODE: DataPooler(), self.VALID_MODE: DataPooler()}
        self.valuewatcher = ValueWatcher()
        self.epochs = 100
        self.best_results = {}
        self.hyper_params = factory.hyper_params
        self.start_date, self.start_date_for_path = StartDate().get_instance()
        self.study_name = study_name

    def run(self):
        best_score = 0.0
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

                self.predict_and_pool(mode=mode, outputs=outputs, x_batch=x_batch, y_batch=y_batch, e=e)

            metrics = get_metrics(self.poolers[mode].get('epoch{}-predicted_label'.format(e + 1)),
                                  self.poolers[mode].get('epoch{}-y'.format(e + 1)))
            self.poolers[mode].set('epoch{}-metrics'.format(e + 1), metrics)
            self.poolers[mode].set('epoch{}-train_loss'.format(e + 1), running_loss[self.TRAIN_MODE])
            self.poolers[mode].set('epoch{}-valid_loss'.format(e + 1), running_loss[self.VALID_MODE])
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

                    self.predict_and_pool(mode=mode, outputs=outputs, x_batch=x_batch, y_batch=y_batch, e=e)

            metrics = get_metrics(self.poolers[mode].get('epoch{}-predicted_label'.format(e + 1)),
                                  self.poolers[mode].get('epoch{}-y'.format(e + 1)))
            self.poolers[mode].set('epoch{}-metrics'.format(e + 1), metrics)
            self.poolers[mode].set('epoch{}-train_loss'.format(e + 1), running_loss[self.TRAIN_MODE])
            self.poolers[mode].set('epoch{}-valid_loss'.format(e + 1), running_loss[self.VALID_MODE])
            self.valuewatcher.update(metrics['f1'])
            self.batchers[mode].reset(with_shuffle=False)

            now_dt, now_dt_for_path = get_now()
            text_line = '|'.join(['{} {}: {:.3f}'.format(mode, key, 100 * metrics[key])
                                  if key not in {'tp', 'fp', 'fn', 'tn'}
                                  else '{} {}: {}'.format(mode, key, metrics[key])
                                  for key in get_print_keys()])
            print('{}|epoch: {:3d}|train loss: {:.2f}|valid loss: {:.2f}|{}'.format(now_dt,
                                                                                    e + 1,
                                                                                    running_loss[self.TRAIN_MODE],
                                                                                    running_loss[self.VALID_MODE],
                                                                                    text_line))

            if self.valuewatcher.is_updated() or e == 0:
                self.best_results = metrics
                self.best_results['date'] = now_dt
                self.best_results['epoch'] = e + 1
                self.best_results['train_loss'] = running_loss[self.TRAIN_MODE]
                self.best_results['valid_loss'] = running_loss[self.VALID_MODE]
                best_score = self.best_results['f1']
                torch.save(self.model.state_dict(),
                           get_save_model_path(dir_tag=self.start_date_for_path, file_tag='{:.6f}-{:03}-{}'.format(best_score, e + 1, now_dt_for_path)))
            if self.valuewatcher.is_over():
                break

        self.export_results()
        self.export_details()

        return best_score

    def predict_and_pool(self, mode, outputs, x_batch, y_batch, e):
        preds, predicted_label = self.predict(outputs)

        self.poolers[mode].set('epoch{}-x'.format(e + 1), x_batch)
        self.poolers[mode].set('epoch{}-y'.format(e + 1), y_batch)
        self.poolers[mode].set('epoch{}-logits'.format(e + 1), outputs.squeeze().tolist())
        self.poolers[mode].set('epoch{}-preds'.format(e + 1), preds.squeeze().tolist())
        self.poolers[mode].set('epoch{}-predicted_label'.format(e + 1), predicted_label)

    def predict(self, outputs):
        if self.is_binary_task():
            preds = torch.sigmoid(outputs)
            predicted_label = [0 if item < 0.5 else 1 for item in preds.squeeze().tolist()]
        else:
            preds = torch.softmax(outputs, dim=1)
            predicted_label = torch.argmax(preds, dim=1).squeeze().tolist()
        return preds, predicted_label

    def is_binary_task(self):
        if self.hyper_params['num_class'] == 1:
            return True
        else:
            return False

    def to_string_hyper_params(self):
        hyper_params = '|'.join(
            ['{}:{}'.format(key, self.hyper_params[key]) if type(self.hyper_params[key]) != list else '{}:{}'.format(
                key, '/'.join(self.hyper_params[key])) for key in sorted(self.hyper_params.keys(), reverse=True)])
        return hyper_params

    def export_results(self):
        path = get_results_path(self.study_name)
        while True:
            try:
                with path.open('a', encoding='utf-8-sig') as f:
                    hyper_params = self.to_string_hyper_params()
                    f.write(','.join([os.path.basename(__file__)]
                                     + [str(self.best_results[key]) for key in ['date', 'epoch', 'train_loss',
                                                                                'valid_loss'] + get_print_keys()]
                                     + [hyper_params]))
                    f.write('\n')
                    break
            except IOError:
                time.sleep(0.5)

    def export_details(self):
        for mode in [self.TRAIN_MODE, self.VALID_MODE]:
            e = 'epoch{}'.format(self.best_results['epoch'])
            xs = self.poolers[mode].get(e + '-x')
            ys = self.poolers[mode].get(e + '-y')
            logits = self.poolers[mode].get(e + '-logits')
            preds = self.poolers[mode].get(e + '-preds')
            predicted_labels = self.poolers[mode].get(e + '-predicted_label')
            with get_details_path(tag='{}-{}-{}-{}'.format(self.start_date_for_path, self.study_name, e, mode))\
                    .open('w', encoding='utf-8-sig') as f:
                for x, y, logit, pred, plabel in zip(xs, ys, logits, preds, predicted_labels):
                    f.write('\t'.join(list(map(str, [x, y, logit, pred, plabel]))))
                    f.write('\n')

        while True:
            try:
                with get_details_path(tag='summary').open('a', encoding='utf-8-sig') as f:
                    hyper_params = self.to_string_hyper_params()
                    f.write(
                        ','.join(
                            [os.path.basename(__file__), self.start_date]
                            + [str(self.best_results[key]) for key in ['date', 'epoch', 'train_loss',
                                                                       'valid_loss'] + get_print_keys()]
                            + [hyper_params]))
                    f.write('\n')
                    break
            except IOError:
                time.sleep(0.5)


if __name__ == '__main__':
    runner = Runner(device='cuda:0')
    score = runner.run()
