import numpy as np
from DP_timegan import DP_Timegan
from timegan import Timegan
from metrics.discriminative_metric import discriminative_score_metrics
from metrics.predictive_metric import predictive_score_metrics
import metrics.plotting_and_visualization as pv
import metrics.oneclass_evaluation as oneclass
import metrics.downstream_model as downstream
import logging
import torch_utils as tu
import torch
import os
import gc
import matplotlib.pyplot as plt
from opacus.utils.batch_memory_manager import BatchMemoryManager

def train(ori_data, parameters, checkpoint_file):
    """
    Train the TimeGAN model using a three-phase training process: 
    embedding network training, supervised loss training, and joint training.

    Args:
        - ori_data (np.ndarray): The original training data.
        - parameters (dict): The options/parameters for training, including the number of iterations.
        - checkpoint_file (str): The file path for saving the model checkpoints.
    """
    model = Timegan(ori_data, parameters, checkpoint_file)

    # 1. Embedding network training
    print('Start Embedding Network Training')
    for i in range(model.start_epoch['embedding'], parameters["iterations"]):
        Etotal_loss, Ebatch_count = 0, 0
        for X_batch in model.dataloader:
            model.X = X_batch.float().to(model.device)
            model.gen_batch()
            model.forward_embedder_recovery()
            model.train_embedder()

            Etotal_loss += np.sqrt(model.E_loss_T0.item())
            Ebatch_count += 1
        Eaverage_loss = Etotal_loss / Ebatch_count
        
        if (i+1) % parameters["print_times"] == 0:
            print(f'step: {str(i+1)}/{str(parameters["iterations"])}, average_e_loss: {str(np.round(Eaverage_loss, 4))}')
        
        if (i+1) % 1000 == 0 or i==(parameters["iterations"]-1):
            phase1_epochnum = {'embedding': i+1, 'supervisor': 0, 'joint': 0}
            model.save_checkpoint(phase1_epochnum, checkpoint_file)

    print('Finish Embedding Network Training')

    # 2. Training only with supervised loss
    print('Start Training with Supervised Loss Only')
    for i in range(model.start_epoch['supervisor'], parameters["iterations"]):
        Stotal_loss, Sbatch_count = 0, 0
        for X_batch in model.dataloader:
            model.X = X_batch.float().to(model.device)
            model.gen_batch()
            model.forward_supervisor()
            model.train_supervisor()

            Stotal_loss += np.sqrt(model.G_loss_S.item())
            Sbatch_count += 1
        Saverage_loss = Stotal_loss / Sbatch_count

        if (i+1) % parameters["print_times"] == 0:
            print(f'step: {str(i+1)}/{str(parameters["iterations"])}, average_s_loss: {str(np.round(Saverage_loss, 4))}')

        
        if (i+1) % 1000 == 0 or i==(parameters["iterations"]-1):
            phase2_epochnum = {'embedding': parameters["iterations"], 'supervisor': i+1, 'joint': 0}
            model.save_checkpoint(phase2_epochnum, checkpoint_file)
    print('Finish Supervised-Only Training')

    # 3. Joint Training
    print('Start Joint Training')
    for i in range(model.start_epoch['joint'], parameters["iterations"]):
        Gtotal_loss, Dtotal_loss, Jbatch_count = 0, 0, 0
        for X_batch in model.dataloader:
            model.X = X_batch.float().to(model.device)
            for kk in range(2):
                model.gen_batch()
                model.forward_generator_discriminator()
                model.train_generator()
                model.forward_embedder_recovery()
                model.train_embedder()
            # # Discriminator training
            model.gen_batch()
            model.forward_generator_discriminator()
            model.train_discriminator()
            
            Gtotal_loss += model.G_loss.item()
            Dtotal_loss += model.D_loss.item()
            Jbatch_count += 1
        Gaverage_loss = Gtotal_loss / Jbatch_count if Jbatch_count > 0 else 0
        Daverage_loss = Dtotal_loss / Jbatch_count if Jbatch_count > 0 else 0

        # Print multiple checkpoints
        if (i+1) % (parameters["print_times"]/10) == 0:
            print(
                f'step: {i+1}/{parameters["iterations"]}, '
                f'd_loss: {np.round(Daverage_loss, 4)}, '
                f'g_loss: {np.round(Gaverage_loss, 4)}, '
                #f'g_loss_u: {np.round(model.G_loss_U.item(), 4)}, '
                #f'g_loss_v: {np.round(model.G_loss_V.item(), 4)}, '
                #f'e_loss_t0: {np.round(np.sqrt(model.E_loss_T0.item()), 4)}'
            )

        if (i+1) % 1000 == 0 or i==(parameters["iterations"]-1):
            phase3_epochnum = {'embedding': parameters["iterations"], 'supervisor': parameters["iterations"], 'joint': i+1}
            model.save_checkpoint(phase3_epochnum, checkpoint_file)
    print('Finish Joint Training')

def dp_train(ori_data, parameters, checkpoint_file, delta=1e-5):
    """
    Train the TimeGAN model using a three-phase training process: 
    embedding network training, supervised loss training, and joint training.

    Args:
        - ori_data (np.ndarray): The original training data.
        - parameters (dict): The options/parameters for training, including the number of iterations.
        - checkpoint_file (str): The file path for saving the model checkpoints.
    """
    model = DP_Timegan(ori_data, parameters, checkpoint_file)

    # 1. Embedding network training
    print('Start Embedding Network Training')
    for i in range(model.start_epoch['embedding'], parameters["iterations"]):
        model.gen_batch()
        #model.batch_forward()
        model.forward_embedder_recovery()
        model.train_embedder()

        if (i+1) % parameters["print_times"] == 0:
            print(f'step: {str(i+1)}/{str(parameters["iterations"])}, e_loss: {str(np.round(np.sqrt(model.E_loss_T0.item()), 4))}')
            print(f"Memory Allocated: {torch.cuda.memory_allocated(model.device) / 1024**2} MB")
        
        if (i+1) % 1000 == 0 or i==(parameters["iterations"]-1):
            phase1_epochnum = {'embedding': i+1, 'supervisor': 0, 'joint': 0}
            model.save_checkpoint(phase1_epochnum, checkpoint_file)
    print('Finish Embedding Network Training')

    # 2. Training only with supervised loss
    print('Start Training with Supervised Loss Only')
    for i in range(model.start_epoch['supervisor'], parameters["iterations"]):
        model.gen_batch()
        #model.batch_forward()
        model.forward_supervisor()
        model.train_supervisor()
        
        if (i+1) % parameters["print_times"] == 0:
            print(f'step: {str(i+1)}/{str(parameters["iterations"])},  g_loss_s: {str(np.round(np.sqrt(model.G_loss_S.item()), 4))}')
            print(f"Memory Allocated: {torch.cuda.memory_allocated(model.device) / 1024**2} MB")
        
        if (i+1) % 1000 == 0 or i==(parameters["iterations"]-1):
            phase2_epochnum = {'embedding': parameters["iterations"], 'supervisor': i+1, 'joint': 0}
            model.save_checkpoint(phase2_epochnum, checkpoint_file)
    print('Finish Supervised-Only Training')

    #model.reset_gradients()
    model.reinitialize_discriminator()
    model.initialize_privacy_engine()
    #3. Joint Training
    print('Start Joint Training')
    model.discriminator.train()
    model.generator.train()
    for i in range(model.start_epoch['joint'], parameters["iterations"]):
        # with BatchMemoryManager(
        #     data_loader=model.dataloader, max_physical_batch_size=128, optimizer=model.optim_discriminator
        # ) as model.dataloader:
        for X_batch, _ in model.dataloader:
            X_batch = X_batch.to(model.device)
            model.train_joint(X_batch)

        # Print multiple checkpoints
        if (i+1) % (parameters["print_times"] / 10) == 0:
            print(
                f'step: {i+1}/{parameters["iterations"]}, '
                f'd_loss: {np.round(model.D_loss.item(), 4)}, '
                f'g_loss_u: {np.round(model.G_loss.item(), 4)}, '
                f'g_loss_v: {np.round(model.G_loss_V.item(), 4)}, '
                #f'e_loss_t0: {np.round(np.sqrt(model.E_loss_T0.item()), 4)}'
            )
            print(f"Memory Allocated: {torch.cuda.memory_allocated(model.device) / 1024**2} MB")

        if (i+1) % 1000 == 0 or i==(parameters["iterations"]-1):
            phase3_epochnum = {'embedding': parameters["iterations"], 'supervisor': parameters["iterations"], 'joint': i+1}
            model.save_checkpoint(phase3_epochnum, checkpoint_file)

            epsilon = model.privacy_engine.get_epsilon(delta=1e-5)
            print(f"(ε = {epsilon:.2f}, δ = {1e-5})")
    print('Finish Joint Training')
    return {'epsilon': parameters["eps"], 'delta': delta}

def test(ori_data, labels, parameters, filename, privacy_params=None):
    """Test the TimeGAN model by generating synthetic data and evaluating its performance using discriminative 
    and predictive scores, followed by visualization using PCA and t-SNE.
    ----------Inputs-----------
        ori_data (np.ndarray): The original data used for generating synthetic data and evaluation.
        parameters (dict): The options/parameters for testing, including synthetic data size and metric iterations.
        filename (str): The file path for saving the visualization output.
    """
    ori_data = np.asarray(ori_data)
    print('Start Testing')
    if parameters["use_dp"]:
        model = DP_Timegan(ori_data, parameters, filename)
        parameters["model_name"] = "DP-TimeGAN"
        print("Loading in DP-TimeGAN")
    else:
        model = Timegan(ori_data, parameters, filename)
        parameters["model_name"] = "TimeGAN"
        print("Loading in TimeGAN")

    # Synthetic data generation
    synth_size = min(parameters["synth_size"], len(ori_data))
    generated_data = model.gen_synth_data()
    generated_data = generated_data.cpu().detach().numpy()

    #gen data is still normed for later use, will be denormed before plotting
    gen_data = [generated_data[i, :model.max_seq_len, :] for i in range(synth_size)]
    gen_data = np.asarray(gen_data)

    #denorm
    denorm_gen_data = np.array([(data * model.max_val) + model.min_val for data in gen_data])

    denorm_ori_data = ori_data[:synth_size, :, :]
    ori_data = model.ori_data[:synth_size, :, :]
    labels = labels[:synth_size]
    print('Finish Synthetic Data Generation')

    # Performance metrics
    metric_results = dict()
    predictive_score, discriminative_score = list(), list()
    # 1. Discriminative Score
    print('Start discriminative score metrics')
    for i in range(parameters["metric_iteration"]):
        print('discriminative score iteration: ', i + 1)
        temp_disc = discriminative_score_metrics(denorm_ori_data, denorm_gen_data)
        discriminative_score.append(temp_disc)
    metric_results['discriminative'] = np.mean(discriminative_score)
    print('Finish discriminative score metrics compute')

    # 2. Predictive score
    print('Start predictive score metrics')
    for i in range(parameters["metric_iteration"]):
        print('predictive score iteration: ', i + 1)
        temp_predict = predictive_score_metrics(denorm_ori_data, denorm_gen_data)
        predictive_score.append(temp_predict)
    metric_results['predictive'] = np.mean(predictive_score)
    print('Finish predictive score metrics compute')

    # 3. Downstream model accuracy
    print('Start downstream model metric')
    if parameters["metric_iteration"] > 0 and parameters["data_name"] != "sines":
        downstream_accuracy = downstream.downstream_model_metrics(gen_data, labels, parameters)
        metric_results["downstream_accuracy"] = downstream_accuracy

    # 4. OneClass metrics
    print('Start OneClass metrics')
    oneclass_metrics_results = {}
    if parameters["metric_iteration"] > 0:
        oneclass_metrics_results = oneclass.evaluate_oneclass_metrics(denorm_ori_data, denorm_gen_data, parameters)
    print('Finish OneClass metrics')

    tu.save_results_to_excel(f'{filename}', metric_results, oneclass_metrics_results, parameters)

    log_path = "./results.log"
    log_path = os.path.abspath(log_path)

    if not os.path.exists(log_path):
        open(log_path, "a").close()

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )
    logging.info(f"Model type: {parameters['model_name']}")
    logging.info(f"Random Seed: {parameters['seed']}")
    logging.info(f'Utility Results: {metric_results}')
    logging.info(f'OneClass Metrics Results: {oneclass_metrics_results}')
    logging.info(f'Iterations: {parameters["iterations"]}, Data Name: {parameters["data_name"]}, DP Enabled: {parameters["use_dp"]}')
    if parameters["use_dp"] and privacy_params:
        epsilon = privacy_params.get("epsilon", "N/A")
        delta = privacy_params.get("delta", 1e-5)
        logging.info(f"(ε = {epsilon:.2f}, δ = {delta})")
    
    logging.info("=" * 100)
    logging.shutdown()

    # 3. Visualization (Original versus Generated, PCA and tSNE)
    fig = pv.plot_4pane(denorm_ori_data, denorm_gen_data, filename=f'{filename}', parameters=parameters)
    plt.close(fig)

    del model, ori_data, denorm_ori_data, gen_data, generated_data, denorm_gen_data
    gc.collect()
    torch.cuda.empty_cache()
