from ray import tune
import os
import shutil
from src.utils import *
import logging

logger = logging.getLogger(__name__)

class CleanupCallback(tune.Callback):
    def on_trial_complete(self, iteration, trials, trial, **info):
        logger.info(f"Cleaning up trial {trial.trial_id}")
        logger.info(f"Trial path: {trial.path}")
        trials_current_best = max(
            [trial.metric_analysis['eval_recall']['max'] for trial in trials if
                'eval_recall' in trial.metric_analysis.keys()])
        for trial in trials:
            if trial.status == 'TERMINATED':
                if trials_current_best > trial.metric_analysis['eval_recall']['max']:
                    self.cleanup_trial(trial)
                else:
                    # Make sure it did all the iterations:
                    # assert trial.last_result['training_iteration'] == 10
                    logger.info(
                        f"Current best: {trial.trial_id} with eval_recall : {trial.metric_analysis['eval_recall']['max']} at iteration {trial.last_result['training_iteration']}")

    def cleanup_trial(self, trial):
        # cleaning up all the /tmp models saved for this trial
        logger.info(f"Cleaning up trial {trial.trial_id}")
        logger.info(f"Trial path: {trial.path}")
        if os.path.exists(trial.path):
            checkpoint_dir = [d for d in os.listdir(trial.path) if 'checkpoint' in d]
            for d in checkpoint_dir:
                shutil.rmtree(trial.path + '/' + d)

        tmp_models = '/'.join(trial.local_experiment_path.split('/')[:-1]) + '/working_dirs'
        tmp_models += os.listdir(tmp_models)[0] + '/run-' + str(trial.trial_id)
        if os.path.exists(tmp_models):
            shutil.rmtree(tmp_models)

class MyCallback(tune.Callback):
    def on_trial_complete(self, iteration, trials, trial, **info):
        logger.info(f"Trial metric analysis: {trial.metric_analysis}")
        return super().on_trial_complete(iteration, trials, trial, **info)
