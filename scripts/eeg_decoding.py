import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import logging, loading, regression, tools, stats
import torch
from pathlib import Path


class eegDecoding:
    def __init__(self, args):
        self.process = 'eegDecoding'
        logging.neptune_init(self.process)
        self.annotations_file = args.annotations_file
        self.eeg_file = args.eeg_file
        self.out_dir = args.out_dir
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.neptune_params(vars(self))

    @staticmethod
    def score_results(y_hat, y_test):
        return tools.to_numpy(stats.corr2d_gpu(y_hat, y_test))

    def load_split_norm(self):
        annotation_raw = loading.load_annotations(self.annotations_file)
        eeg_raw = loading.load_eeg(self.eeg_file)
        eeg_filtered, annotations = loading.check_videos(eeg_raw, annotation_raw)
        eeg = loading.strip_eeg(eeg_filtered)
        y_train, y_test, X_train, X_test = regression.split_data(annotations, eeg)
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

    def viz_results(self, scores):
        fig, ax = plt.subplots()
        ax.hist(scores)
        logging.neptune_results(fig)

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
        self.viz_results(results['scores'])
        logging.neptune_stop()


def main():
    parser = argparse.ArgumentParser(description='Predict fMRI responses using the features')
    parser.add_argument('--annotations_file', '-a', type=str, help='annotations file path')
    parser.add_argument('--eeg_file', '-e', type=str, help='preprocessed EEG file')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs')
    parser.add_argument('--alpha_start', type=int, default=-5,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=2,
                        help='stopping value in log space for the ridge alpha penalty')
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. see DeepJuice TorchRidgeGV for options')
    args = parser.parse_args()
    eegDecoding(args).run()


if __name__ == '__main__':
    main()