import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import logging, loading, regression
import torch


class fmriEncoding:
    def __init__(self, args):
        self.process = 'fmriEncoding'
        logging.neptune_init(self.process)
        self.annotations_file = args.annotations_file
        self.fmri_dir = args.fmri_dir
        self.out_file = args.out_file
        self.alpha_start = args.alpha_start
        self.alpha_stop = args.alpha_stop
        self.scoring = args.scoring
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.neptune_params(vars(self))

    @staticmethod
    def score_results(y_hat, y_test):
        return tools.to_numpy(stats.corr2d_gpu(y_hat, y_test))

    def load_split_norm(self):
        annotations = loading.load_annotations(self.annotations_file)
        fmri, _ = loading.load_fmri(self.fmri_dir)
        X_train, X_test, y_train, y_test = regression.split_data(annotations, fmri)
        regression.preprocess(X_train, X_test, y_train, y_test)
        return tools.to_torch([X_train, X_test, y_train, y_test],
                              device=self.device)

    def save_results(self, scores):
        pd.DataFrame(scores).to_csv(self.out_file, index=False)

    def viz_results(self, scores):
        fig = plt.hist(scores)
        logging.neptune_results(fig)

    def run(self):
        [X_train, X_test, y_train, y_test] = self.load_split_norm()
        y_hat = regression.regress_and_predict(X_train, X_test, y_train,
                                               alpha_start=self.alpha_start,
                                               alpha_stop=self.alpha_stop,
                                               scoring=self.scoring,
                                               device=self.device)
        scores = score_results(y_hat, y_test)
        self.save_results(scores)
        self.viz_results(scores)
        logging.neptune_stop()
        

def main():
    parser = argparse.ArgumentParser(description='Predict fMRI responses using the features')
    parser.add_argument('--annotations_file', '-a', type=str, help='annotations file path')
    parser.add_argument('--fmri_dir', '-f', type=str, help='fMRI benchmarks directory')
    parser.add_argument('--out_file', '-o', type=str, help='output file for the regression results')
    parser.add_argument('--alpha_start', type=int, default=-2,
                        help='starting value in log space for the ridge alpha penalty')
    parser.add_argument('--alpha_stop', type=int, default=5,
                        help='stopping value in log space for the ridge alpha penalty')
    parser.add_argument('--scoring', type=str, default='pearsonr',
                        help='scoring function. see DeepJuice TorchRidgeGV for options')
    args = parser.parse_args()
    fmriEncoding(args).run()


if __name__ == '__main__':
    main()