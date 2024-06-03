from src import tools

tools.neptune_init('test')
tools.neptune_params({'date': 'now'})

tools.neptune_results('/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/reports/figures/FeatureDecoding/sub-03_reg-gaze-False_decoding.png')
tools.neptune_stop()
