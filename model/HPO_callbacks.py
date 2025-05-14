from ray import tune
import os
import shutil

class CleanupCallback(tune.Callback):
            def on_trial_complete(self, iteration, trials, trial, **info):
                trials_current_best = max(
                    [trial.metric_analysis['eval_' + args.metrics][args.direction[0]] for trial in trials if
                     'eval_' + args.metrics in trial.metric_analysis.keys()])
                for trial in trials:
                    if trial.status == 'TERMINATED':
                        if trials_current_best > trial.metric_analysis['eval_' + args.metrics][args.direction[0]]:
                            self.cleanup_trial(trial)
                        else:
                            # Make sure it did all the iterations:
                            # assert trial.last_result['training_iteration'] == 10
                            print(
                                f"Current best: {trial.trial_id} with eval_{args.metrics}: {trial.metric_analysis['eval_' + args.metrics][args.direction[0]]} at iteration {trial.last_result['training_iteration']}")
 
            def cleanup_trial(self, trial):
                # clearning up all the /tmp models saved for this trial
                print(f"Cleaning up trial {trial.trial_id}")
                if os.path.exists(trial.path):
                    checkpoint_dir = [d for d in os.listdir(trial.path) if 'checkpoint' in d]
                    for d in checkpoint_dir:
                        shutil.rmtree(trial.path + '/' + d)
 
                tmp_models = '/'.join(trial.local_experiment_path.split('/')[:-1]) + '/working_dirs/save_models/'
                tmp_models += os.listdir(tmp_models)[0] + '/run-' + str(trial.trial_id)
                if os.path.exists(tmp_models):
                    shutil.rmtree(tmp_models)


""" 
            class MyCallback(tune.Callback):
                def on_trial_start(self, iteration, trials, trial, **info):
                    logger.info(f"Trial successfully started with config : {trial.config}")
                    return super().on_trial_start(iteration, trials, trial, **info)
                
                def on_trial_complete(self, iteration, trials, trial, **info):
                    logger.info(f"Trial ended with config : {trial.config}")
                    return super().on_trial_complete(iteration, trials, trial, **info)
                
                def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
                    logger.info("Created checkpoint successfully")
                    return super().on_checkpoint(iteration, trials, trial, checkpoint, **info)
            """