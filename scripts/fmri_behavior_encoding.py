import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import logging, loading, regression, tools, stats
import torch
from pathlib import Path


class fmriBehaviorEncoding:
    def __init__(self, args):
        self.process = 'fmriBehaviorEncoding'
        self.fmri_dir = args.fmri_dir
        self.out_dir = args.out_dir
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def score_results(y_hat, y_test):
        return stats.corr2d_gpu(y_hat, y_test)

    def load_split_norm(self):
        annotations = loading.load_behavior(self.fmri_dir)
        fmri, _ = loading.load_fmri(self.fmri_dir)
        eeg = loading.strip_eeg(eeg_filtered)
        splits = regression.split_data(annotations, {'fmri': fmri})
        X_train, X_test = splits['fmri_train'], splits['fmri_test']
        y_train, y_test = splits['annotations_train'], splits['annotations_test']
        [X_train, X_test, y_train, y_test] = tools.to_torch([X_train, X_test, y_train, y_test],
                                                            device=self.device)
        regression.preprocess(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test

    def save_results(self, results):
        for key, val in results.items():
            out_file = f'{self.out_dir}/{key}.csv.gz'
            pd.DataFrame(tools.to_numpy(val)).to_csv(out_file, index=False)

    def mk_out_dir(self):
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)

    def run(self):
        [X_train, X_test, y_train, y_test] = self.load_split_norm()
        results = regression.regress_and_predict(X_train, y_train, X_test,
                                                 alpha_start=self.alpha_start,
                                                 alpha_stop=self.alpha_stop,
                                                 scoring=self.scoring,
                                                 device=self.device)
        results['scores'] = self.score_results(results['yhat'], y_test)
        self.mk_out_dir()
        self.save_results(results)


def main():
    parser = argparse.ArgumentParser(description='Predict fMRI responses using the features')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory')
    parser.add_argument('--out_dir', '-o', type=str, help='output directory for the regression results')
    parser.add_argument('--alpha_start', type=int, default=-5,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=2,
                        help='stopping value in log space for the ridge alpha penalty')
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. see DeepJuice TorchRidgeGV for options')
    args = parser.parse_args()
    fmriBehaviorEncoding(args).run()


if __name__ == '__main__':
    main()