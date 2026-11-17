'''
TODO List:
    
'''
import os, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import optuna
import copy, json
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn.model_selection import KFold, train_test_split
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', 
                    filename='logs.log', encoding='utf-8', level=logging.DEBUG)
plt.set_loglevel('warning')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_database(filename):
    df = pd.read_csv(filename)
    logging.info(f'{filename} read')
    return df

def read_all_databases(namelist):
    dflist = []
    for filename in namelist:
        dflist.append(read_database(filename))
    stacked = pd.concat(dflist, axis=0)
    return stacked

def plot_cross_sections(
    df, Z_ion, Z_target,
    show_uncertainty=True,   # draw error bars for experiments if available
    exp_marker='o',
    theory_linestyle='-'
):
    # Filter by selected projectile-target pair
    df_sub = df[(df['Z_ion'] == Z_ion) & (df['Z_target'] == Z_target)].copy()
    if df_sub.empty:
        raise ValueError(f"No rows for Z_ion={Z_ion}, Z_target={Z_target}")

    # Ensure numeric types
    df_sub['E_proj[keV/u]'] = pd.to_numeric(df_sub['E_proj[keV/u]'], errors='coerce')
    df_sub['Cross_section[cm2]'] = pd.to_numeric(df_sub['Cross_section[cm2]'], errors='coerce')
    if 'Uncertainty[%]' in df_sub.columns:
        df_sub['Uncertainty[%]'] = pd.to_numeric(df_sub['Uncertainty[%]'], errors='coerce')

    # Group by Theory_ID while preserving input order of groups
    theory_ids = df_sub['Theory_ID'].drop_duplicates().tolist()

    plt.figure(figsize=(7.2, 4.6))

    for theory in theory_ids:
        df_model = df_sub[df_sub['Theory_ID'] == theory].sort_values('E_proj[keV/u]')

        # Decide if this block is experimental
        if 'is_exp' in df_model.columns:
            is_experimental = (df_model['is_exp'].astype(int) == 1).any()
        else:
            is_experimental = ('exp' in str(theory).lower())

        # Style by type
        if is_experimental:
            # Experimental: markers, no connecting line
            if show_uncertainty and 'Uncertainty[%]' in df_model.columns:
                yerr = df_model['Cross_section[cm2]'] * (df_model['Uncertainty[%]'].fillna(0.0) / 100.0)
                plt.errorbar(
                    df_model['E_proj[keV/u]'],
                    df_model['Cross_section[cm2]'],
                    yerr=yerr,
                    fmt=exp_marker,
                    linestyle='None',
                    label=f"{theory} (Exp)",
                    alpha=1.0,
                    capsize=2,
                    zorder=3
                )
            else:
                plt.plot(
                    df_model['E_proj[keV/u]'],
                    df_model['Cross_section[cm2]'],
                    linestyle='None',
                    marker=exp_marker,
                    label=f"{theory} (Exp)",
                    alpha=1.0,
                    zorder=3
                )
        else:
            # Theory: continuous line
            plt.plot(
                df_model['E_proj[keV/u]'],
                df_model['Cross_section[cm2]'],
                linestyle=theory_linestyle,
                marker=None,
                label=theory,
                alpha=0.9,
                zorder=2
            )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Projectile Energy (keV/u)')
    plt.ylabel('Total Ionization Cross Section (cm²)')
    plt.title(f'Total Ionization Cross Sections\nProjectile Z={Z_ion}, Target Z={Z_target}')
    plt.grid(True, which='both', ls='--', lw=0.5, alpha=0.7)
    plt.legend(ncol=1, fontsize=9)
    plt.tight_layout()
    plt.savefig('test.png')
    plt.show()

    # --- Define the weights ---

def smooth_step(x, center, width=10):
    """Logistic transition around 'center', width controls smoothness."""
    return 1 / (1 + np.exp(-(x - center) / width))

def assign_weights(df):
    def compute_weight(row):
        theory = row['Theory_ID']
        E = row['E_proj[keV/u]']

        if theory == 'Exp':
            return 2.0
        elif theory == 'CDW-EIS':
            return 1.0 * smooth_step(E, 100, width=10)
        elif theory == 'Born':
            return 0.5 * smooth_step(E, 100, width=10)
        elif theory == 'CTMC':
            return 0.5 * (1 - smooth_step(E, 200, width=10))  # fade out above 200
        else:
            return 0.2 * smooth_step(E, 50, width=20)  # low reliability, mild fade-in

    weights = df.apply(compute_weight, axis=1)
    return weights.values.astype(np.float32)

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Compute the difference between prediction and true values
        diff = y_pred - y_true
        
        # Compute the log-cosh loss
        loss = torch.mean(torch.log(torch.cosh(diff)))
        
        return loss

def combined_loss(pred, target, alpha=0.7):
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    return alpha * mse + (1 - alpha) * mae

def small_mid_huge_loss(pred, target):
    huge = torch.nn.MSELoss()
    mid  = torch.nn.L1Loss()
    small = torch.mean(torch.abs((pred - target)/target))
    return huge(target, pred) + mid(target, pred) + small

class GaussianActivation(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.exp(-x**2)

class FlexibleNN(torch.nn.Module):
    def __init__(self, input_dim, n_hidden, activations, dropout_rate):
        super().__init__()
        modules = []

        self.lin1 = torch.nn.Linear(input_dim, n_hidden)
        self.lin2 = torch.nn.Linear(n_hidden,n_hidden)
        # self.lin3 = torch.nn.Linear(n_hidden,n_hidden)
        # self.lin4 = torch.nn.Linear(n_hidden,n_hidden)
        self.lin5 = torch.nn.Linear(n_hidden, 1)
        if activations == 'tanh':
            self.act = torch.nn.Tanh()
            logging.info(f'using activation tanh')
        elif activations == 'relu':
            self.act = torch.nn.ReLU()
            logging.info(f'using activation ReLU')
        elif activations == 'elu':
            self.act = torch.nn.ELU()
            logging.info(f'using activation ELU')
        elif activations == 'logsigmoid':
            self.act = torch.nn.LogSigmoid()
            logging.info(f'using activation LogSigmoid')
        elif activations == 'softplus':
            self.act = torch.nn.Softplus()
            logging.info(f'using activation Softplus')
        elif activations == 'gauss':
            self.act = GaussianActivation()

        self.dp = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.dp(x)
        x = self.act(self.lin2(x))
        x = self.dp(x)
        # x = self.act(self.lin3(x))
        # x = self.dp(x)
        # x = self.act(self.lin4(x))
        # x = self.dp(x)
        return self.lin5(x)

def objective_conditional(trial, tensors):
    # n_epochs = trial.suggest_categorical("epochs", 500, 1000])
    # n_layers = trial.suggest_int("n_layers", 1, 3, step=1)
    # hidden_dims = [trial.suggest_categorical(f"n_units_l{i}", [16, 32, 64]) for i in range(n_layers)]
    # activations = [trial.suggest_categorical(f"act_l{i}", ["relu", "tanh", "leaky_relu"]) for i in range(n_layers)]
    activations = trial.suggest_categorical('activations',['tanh','gauss'])
    # dropout_rate = trial.suggest_float("dropout", 0.05, 0.25)
    # lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    # alpha_loss = trial.suggest_float("alpha_loss", 0.0, 1.0)

    try:
        X_tensor = tensors['x']
        y_tensor = tensors['y']
    except NameError:
        raise RuntimeError("X_conditional and/or y_conditional not defined.")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_validation_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
        model = FlexibleNN(X_tensor.shape[1], 3, activations, 0.05).to(DEVICE)

        # if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # elif optimizer_name == "RMSprop":
        #     optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        # else:
        #     optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        criterion = LogCoshLoss()
        for epoch in range(1000):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_x)
                # loss = combined_loss(predictions, batch_y, alpha_loss)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_tensor[val_idx])
            fold_val_loss = criterion(val_predictions, y_tensor[val_idx]).item()
            fold_validation_losses.append(fold_val_loss)

        trial.report(fold_val_loss, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_validation_losses)

def train_kfolded(tensors, arch):
    X_tensor = tensors['x']
    y_tensor = tensors['y']
    epochs = arch['epochs']
    dp = arch['dropout']
    kf = KFold(n_splits=5, shuffle=True, random_state=12)

    history_train={}
    history_val={}
    metrics_slant = {
        'train':{
            'MAE' : [],
            'RMSE': [],
            'MBE' : [],
            'MRE' : [],
            'MaxAE': [],
            'MaxRE':[]
        },
        'val':{
            'MAE' : [],
            'RMSE': [],
            'MBE' : [],
            'MRE' : [],
            'MaxAE': [],
            'MaxRE':[]
        }}
    metrics = {}
    best_eval_loss = torch.inf
    best_model_idx = 0

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
        history_train[fold_idx] = history_train.get(fold_idx,[])
        history_val[fold_idx] = history_val.get(fold_idx,[])
        metrics[fold_idx]= copy.deepcopy(metrics_slant)

        model = FlexibleNN(X_tensor.shape[1], 
                           arch['n_hidden'], 
                           arch['activation'], 
                           dp).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        if arch['criterion'] == 'logcosh':
            criterion = LogCoshLoss()
        elif arch['criterion'] == 'small_mid_huge':
            criterion = small_mid_huge_loss
        else:
            print('Criterion error')
            exit()
        # Train stage
        for epoch in range(epochs):
            model.train()
            accumulated_loss = []
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
            
                accumulated_loss.append( loss.item())
                loss.backward()
                optimizer.step()

            if epoch%arch['save_steps'] == 0:
                history_train[fold_idx].append(np.sum(accumulated_loss)/len(train_loader))
                metrics[fold_idx]['train']['MAE'].append(MAE(model(X_tensor[train_idx]), y_tensor[train_idx]))
                metrics[fold_idx]['train']['RMSE'].append(RMSE(model(X_tensor[train_idx]), y_tensor[train_idx]))
                metrics[fold_idx]['train']['MBE'].append(MBE(model(X_tensor[train_idx]), y_tensor[train_idx]))
                metrics[fold_idx]['train']['MRE'].append(MRE(model(X_tensor[train_idx]), y_tensor[train_idx]))
                metrics[fold_idx]['train']['MaxAE'].append(MaxAE(model(X_tensor[train_idx]), y_tensor[train_idx]))
                metrics[fold_idx]['train']['MaxRE'].append(MaxRE(model(X_tensor[train_idx]), y_tensor[train_idx]))
                
        # 4. Evaluación del fold
            accumulated_loss = []
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)
                    accumulated_loss.append( loss.item())

                if epoch%arch['save_steps'] == 0:
                    history_val[fold_idx].append(np.sum(accumulated_loss)/len(val_loader))
                    metrics[fold_idx]['val']['MAE'].append(MAE(model(X_tensor[val_idx]), y_tensor[val_idx]))
                    metrics[fold_idx]['val']['RMSE'].append(RMSE(model(X_tensor[val_idx]), y_tensor[val_idx]))
                    metrics[fold_idx]['val']['MBE'].append(MBE(model(X_tensor[val_idx]), y_tensor[val_idx]))
                    metrics[fold_idx]['val']['MRE'].append(MRE(model(X_tensor[val_idx]), y_tensor[val_idx]))
                    metrics[fold_idx]['val']['MaxAE'].append(MaxAE(model(X_tensor[val_idx]), y_tensor[val_idx]))
                    metrics[fold_idx]['val']['MaxRE'].append(MaxRE(model(X_tensor[val_idx]), y_tensor[val_idx]))

                if np.mean(accumulated_loss) < best_eval_loss:
                    best_model = copy.deepcopy(model.state_dict())
                    best_model_idx = fold_idx
                    best_eval_loss = np.sum(accumulated_loss)/len(val_loader)
                
            if epoch%arch['save_steps'] == 0:
                print('------------------------------------------')
                print(f'fold index: {fold_idx}')
                print(f"epoch: {epoch} - train loss: {history_train[fold_idx][-1]:.5f} - val loss: {history_val[fold_idx][-1]:.5f}")


    model.load_state_dict(best_model)
    return model, best_eval_loss, history_train, history_val, metrics

def MAE(predictions, ground_truth):
    return torch.mean(torch.abs(predictions-ground_truth)).item()

def RMSE(predictions, ground_truth):
    return torch.sqrt(torch.mean((predictions-ground_truth)**2)).item()

def MBE(predictions, ground_truth):
    return torch.mean(predictions-ground_truth).item()

def MRE(predictions, ground_truth):
    return torch.mean(torch.abs((predictions-ground_truth)/ground_truth)).item()

def MaxAE(predictions, ground_truth):
    return torch.max(torch.abs(predictions-ground_truth)).item()

def MaxRE(predictions, ground_truth):
    return torch.max(torch.abs((predictions-ground_truth)/ground_truth)).item()

def train_optimum(tensors, arch):
    X_tensor = tensors['x']
    y_tensor = tensors['y']
    epochs = arch['epochs']    

    model = FlexibleNN(X_tensor.shape[1], 
                        arch['n_hidden'], 
                        arch['activation'], 
                        arch['dropout']).to(DEVICE)
    for p in model.named_parameters():
        print(p[0], p[1].shape)
    print(f'model parameters: {sum(p.numel() for p in model.parameters())}')

    history = {'train':[],'val':[]}
    best_model = copy.deepcopy(model.state_dict())
    best_eval_loss = torch.inf

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, 
                                                        y_tensor, 
                                                        test_size=0.2, 
                                                        shuffle=True)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=600, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=600, shuffle=True)
    criterion = small_mid_huge_loss


    for epoch in range(epochs):
        model.train()
        accumulated_loss = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
        
            accumulated_loss.append( loss.item())
            loss.backward()
            optimizer.step()
        history['train'].append(np.mean(accumulated_loss))

        accumulated_loss = []
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                accumulated_loss.append( loss.item())
            history['val'].append(np.mean(accumulated_loss))

            if history['val'][-1] < best_eval_loss:
                best_model = copy.deepcopy(model.state_dict())
                best_eval_loss = history['val'][-1]

        if epoch%100 ==0:
            print('------------------------------------------')
            print(f"epoch: {epoch} - train loss: {history['train'][-1]:.5f} - val loss: {history['val'][-1]:.5f} - best val loss: {best_eval_loss:.5f}")

    model.load_state_dict(best_model)
    return model, history

def pretty_print(info, metrics, filename):
    texto = f'''Architecture:   [6, {info['n_hidden']}, {info['n_hidden']}, {info['n_hidden']}, 1]
        Activation: {info['activation']}
        Epochs: {info['epochs']}
        dropout:  {info['dropout']}
        combined loss small, mid, large
        --- FINAL METRICS [trained over log10(cross section )] ---\n'''
    for k in metrics['train'].keys():
        texto +=f"[Train] best {k}: {metrics['train'][k]:.5f}\n"
        texto +=f"[Test]  best {k}: {metrics['test'][k]:.5f}\n"
    texto += 'best eval loss: '+ str(info['best_loss'])
    with open(file=filename, mode='w') as fp:
        fp.write(texto)


def predict_conditional(model, one_hot_encoder, theory: str, Z_proj, Z_target, values):
    # assertions

    # --- input tensor creation
    enc_array = one_hot_encoder.transform([[theory]])
    input_tensor = torch.hstack([
                  torch.FloatTensor(np.ones_like(values).reshape(-1,1)*Z_proj),
                  torch.FloatTensor(np.ones_like(values).reshape(-1,1)*Z_target),
                  torch.FloatTensor(values.reshape(-1,1)),
                  torch.FloatTensor(np.repeat(enc_array.reshape(1,-1), values.shape[0], axis=0))
                  ]).requires_grad_(False)

    with torch.no_grad():
        predicted = model(input_tensor)
    
    return 10**predicted.numpy()

if __name__ == '__main__':


    # --- Read all databases for training ---
    namelist = [os.path.join('.','dbs','Semiempiric_db', 'semiempiric_TCS_database_single_ionization_atoms_1985Rudd_H+_+_H.csv'),
                os.path.join('.','dbs','ctmc', 'ctmc_TCS_database_single_ionization_hydrogen.csv'),
                os.path.join('.','dbs','CDWEIS', 'cdw-eis_TCS_database_single_ionization_atoms_except_Kr.csv')]

    # namelist = [os.path.join('.','dbs','CDWEIS', 'cdw-eis_TCS_database_single_ionization_atoms_except_Kr.csv')]
        
    df =read_all_databases(namelist)
    logging.info(df.columns)
    # plot_cross_sections(df, 1, 1)
    
    # # --- Prepare the Data ---
    df['log_E'] = np.log10(df['E_proj[keV/u]'].values)
    df['log_sigma'] = np.log10(df['Cross_section[cm2]'].values)

    # One-hot encode Theory_ID

    # --- Data for the Conditional NN with Z_target = 1 ---
    hyd_df = df[df['Z_target']==1]
    print(hyd_df.info)
    encoder = OneHotEncoder(sparse_output=False)
    theory_one_hot = encoder.fit_transform(hyd_df[['Theory_ID']].values)
    
    # joblib.dump(encoder, 'one_hot_encoder_models.joblib')
    df_theory_encoded = pd.DataFrame(theory_one_hot, columns=encoder.get_feature_names_out(['Theory_ID']))
    hyd_df = pd.concat([hyd_df.reset_index(drop=True), df_theory_encoded], axis=1)

    logging.info(hyd_df.info)

    torch.manual_seed(7)
    X_conditional = np.hstack([
        hyd_df[['Z_ion', 'log_E']].values,
        theory_one_hot
    ])
    y_conditional = hyd_df['log_sigma'].values


    train_tensor = {'x':torch.tensor(X_conditional, dtype=torch.float32, device = DEVICE),
        'y': torch.tensor(y_conditional, dtype=torch.float32, device = DEVICE).unsqueeze(1)}
    logging.info('trainset shapes')
    logging.info(f"{train_tensor['x'].shape} , {train_tensor['y'].shape}")
    activations = ['tanh', 'gauss']
    arch = {'epochs':10000}
    arch['save_steps'] = 50
    hidden = [16,32,64,128,256,512]
    arch['dropout'] = 0.001
    crits =  ['logcosh', 'small_mid_huge']
    


    # # -------------- For model selection -------------
    # best_loss = np.inf
    # for criterion in crits:
    #     for nh in hidden:
    #         for act in activations:
    #             arch['n_hidden']   = nh
    #             arch['activation'] = act
    #             arch['criterion'] = criterion
                
    #             model, best_eval_loss, history_train, history_val, metrics = train_kfolded(train_tensor, arch)
    #             print(f'Activation: {act}, best loss:{best_eval_loss}')
    #             print('-------------------------------------')
    #             print('-------------------------------------')
    #             if best_eval_loss < best_loss:
    #                 best_loss = best_eval_loss
    #                 best_model = copy.deepcopy(model.state_dict())
    #                 best_nh = nh
    #                 best_act = act
    #                 best_crit = criterion
    #             with open('metrics_'+str(nh)+'_'+str(nh)+'_'+ act +'_'+criterion+'_indim_5.json', 'w') as fp:
    #                 json.dump(metrics, fp, indent=2)
    #             with open('history_train_'+str(nh)+'_'+str(nh)+'_'+act+'_'+criterion+'_indim_5.json', 'w') as fp:
    #                 json.dump(history_train, fp, indent=2)
    #             with open('history_val_'+str(nh)+'_'+str(nh)+'_'+act+'_'+criterion+'_indim_5.json', 'w') as fp:
    #                 json.dump(history_val, fp, indent=2)
    #     print(f'Best Activation: {best_act}, best loss:{best_loss}, best criterion: {criterion}')
    #     print('-------------------------------------')
    #     print('-------------------------------------')

    # # model = FlexibleNN(train_tensor['x'].shape[1], 
    # #                        arch['n_hidden'], 
    # #                        arch['activation'], 
    # #                        0.001).to(DEVICE)
    # # model.load_state_dict(best_model)
    # # torch.save(model.state_dict(), 'best_statedict_gauss_512.pth')

    # # ---------- End model selection ----------------------

    # ----------- Train the very best configuration

    arch = {'epochs':50000}
    arch['save_steps'] = 50
    arch['dropout'] = 0.001
    arch['n_hidden']   = 128
    arch['activation'] = 'gauss'
    arch['criterion'] = 'small_mid_huge'
    best_model, history= train_optimum(train_tensor, arch)
    torch.save(best_model.state_dict(), 'best_statedict_gauss_128.pth')
    plt.semilogy(history['train'], label = 'train')
    plt.semilogy(history['val'], label = 'val')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel(r'$\mathcal{L}_{\theta}$', fontsize=18)
    plt.legend()

    plt.show()


    # # --- Evaluation stage ---
    # model = FlexibleNN(6, [], "elu", 0.0)
    # model.load_state_dict(torch.load('./modelo_state_dict.pt'))
    # model.eval()
    # cdw_eis = predict_conditional(model, encoder, 'CDW-EIS', 1.0, 1.0, np.log10(np.linspace(10,1000,1500)))
    # ctmc = predict_conditional(model, encoder, 'CTMC', 1.0, 1.0, np.log10(np.linspace(10,1000,1500)))
    # plt.loglog(np.linspace(10,1000,1500),cdw_eis)
    # plt.loglog(np.linspace(10,1000,1500),ctmc)
    # plt.show()


    # # 1. Define a Pruner
    # # MedianPruner: Prunes if the trial's intermediate value is worse than the median of other trials.
    # # - n_startup_trials: Don't prune for the first 5 trials (let Optuna explore a bit first).
    # # - n_warmup_steps: Don't prune before this many steps (folds, in our case) are completed in a trial.
    # #                   Here, we set it to 0, meaning pruning can happen after the first fold (step 0).
    # # - interval_steps: Prune (i.e., check trial.should_prune()) every 'interval_steps' reports.
    # #                   Here, 1 means check after every fold.
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    # # Alternatively, use SuccessiveHalvingPruner:
    # # pruner = optuna.pruners.SuccessiveHalvingPruner()

    # # 2. Create a Study, passing the pruner
    # study = optuna.create_study(direction="minimize", pruner=pruner)

    # # 3. Run optimization
    # # You might want to increase n_trials when using pruning, as some trials will finish early.
    # try:
    #     study.optimize(lambda x : objective_conditional(x, train_tensor), n_trials=100) # Increased n_trials for demonstration
    # except RuntimeError as e:
    #     logging.debug(f"Optimization failed: {e}")
    #     logging.debug("Ensure X_unified and y_unified are properly defined and accessible globally.")

    # # Print results if optimization ran
    # if study.trials: # Check if any trials were run
    #     logging.info("\nStudy statistics: ")
    #     logging.info("  Number of finished trials: ", len(study.trials))
        
    #     pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    #     complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
    #     logging.info("  Number of pruned trials: ", len(pruned_trials))
    #     logging.info("  Number of complete trials: ", len(complete_trials))

    #     if study.best_trial:
    #         logging.info("\nBest trial:")
    #         logging.info("  Value (mean validation loss): ", study.best_trial.value)
    #         logging.info("  Params: ")
    #         for key, value in study.best_trial.params.items():
    #             logging.info(f"    {key}: {value}")
    #     else:
    #         logging.info("\nNo best trial found (e.g., all trials pruned or failed).")
    # else:
    #     logging.info("\nNo trials were run.")
    