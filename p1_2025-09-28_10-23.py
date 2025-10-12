import numpy as np;
import json,os,pathlib,pickle,random,string,time;
from sklearn.preprocessing import StandardScaler;
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold;
#Импорты для классификации:
from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier,Perceptron,RidgeClassifier,SGDClassifier;
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier;
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier;
from xgboost import XGBClassifier;from lightgbm import LGBMClassifier;
from sklearn.metrics import accuracy_score,auc,average_precision_score,balanced_accuracy_score,brier_score_loss;
from sklearn.metrics import cohen_kappa_score,dcg_score,f1_score,fbeta_score,hamming_loss,hinge_loss,jaccard_score;
from sklearn.metrics import log_loss,matthews_corrcoef,ndcg_score,precision_score,recall_score,roc_auc_score,zero_one_loss;
#Импорты для регрессии:
from sklearn.linear_model import LinearRegression,Ridge,SGDRegressor,ElasticNet,Lars,Lasso,LassoLars,LassoLarsIC,ARDRegression;
from sklearn.linear_model import BayesianRidge,MultiTaskElasticNet,MultiTaskLasso,HuberRegressor,QuantileRegressor,RANSACRegressor;
from sklearn.linear_model import TheilSenRegressor,GammaRegressor,PoissonRegressor,TweedieRegressor,PassiveAggressiveRegressor;
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor;
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor;
from sklearn.metrics import d2_absolute_error_score,d2_pinball_score,d2_tweedie_score,explained_variance_score,max_error;
from sklearn.metrics import mean_absolute_percentage_error,mean_gamma_deviance,mean_pinball_loss,mean_poisson_deviance;
from sklearn.metrics import mean_squared_error,mean_squared_log_error,mean_tweedie_deviance,median_absolute_error;
from sklearn.metrics import r2_score,root_mean_squared_error,root_mean_squared_log_error;

def load_data():
    assert os.path.exists('hw_final_open_data.npy'), 'Please, download hw_final_open_data.npy and place it in the working directory'
    assert os.path.exists('hw_final_open_target.npy'), 'Please, download hw_final_open_target.npy and place it in the working directory'
    assert os.path.exists('hw_final_closed_data.npy'), 'Please, download hw_final_closed_data.npy and place it in the working directory'
    opened_data:np.ndarray=np.load('hw_final_open_data.npy', allow_pickle=False)
    print(f'type(opened_data): {type(opened_data)}');
    n_samples_opened:int=opened_data.shape[0];
    n_features_opened:int=opened_data.shape[1];
    print(f'n_samples_opened: {n_samples_opened}, n_features_opened: {n_features_opened}');
    print(f'opened_data.shape: {opened_data.shape}');
    print(f'opened_data[0]: {opened_data[0]}');
    print(f'opened_data[1]: {opened_data[1]}');
    print(f'opened_data[2]: {opened_data[2]}');
    print(f'opened_data[{n_samples_opened-1}]: {opened_data[n_samples_opened-1]}');
    opened_target:np.ndarray=np.load('hw_final_open_target.npy', allow_pickle=False)
    print(f'type(opened_target): {type(opened_target)}');
    print(f'opened_target.shape: {opened_target.shape}');
    print(f'opened_target[0]: {opened_target[0]}');
    print(f'opened_target[1]: {opened_target[1]}');
    print(f'opened_target[2]: {opened_target[2]}');
    print(f'opened_target[{n_samples_opened-1}]: {opened_target[n_samples_opened-1]}');
    closed_data:np.ndarray=np.load('hw_final_closed_data.npy', allow_pickle=False)
    print(f'type(closed_data): {type(closed_data)}');
    n_samples_closed:int=closed_data.shape[0];
    n_features_closed:int=closed_data.shape[1];
    print(f'n_samples_closed: {n_samples_closed}, n_features_closed: {n_features_closed}');
    print(f'closed_data.shape: {closed_data.shape}');
    print(f'closed_data[0]: {closed_data[0]}');
    print(f'closed_data[1]: {closed_data[1]}');
    print(f'closed_data[2]: {closed_data[2]}');
    print(f'closed_data[{closed_data.shape[0]-1}]: {closed_data[closed_data.shape[0]-1]}');
    closed_target:np.ndarray=np.ndarray(shape=(n_samples_closed,),dtype=opened_target.dtype);

    
    #Функция run_one_model_experiment_v3 [тут переименована в run_one_model_experiment_v5] взята из моего кода для задачи F. Биометрия
    #(задача бинарной классификации, определить пол по голосу)
    #[G:\Мой диск\IT (Python SQL DE DS ML)\Резюме и работа (НЕ видео и аудио)\Яндекс стажировка задачи\Задача F Биометрия]
    #Там эта функция отлично себя показала, но тут нужно будет сделать некоторые изменения в коде, так как:
    #1. Там классификация, тут регрессия => другие модели, ансамбли и метрики (MSE или RMSE вместо accuracy)
    #2. Изменение названия переменных:
    #x_13936 -> opened_data [массив значений признаков (features) для открытых данных]
    #y_13936 -> opened_target [массив target значений для открытых данных]
    #id_13936 -> opened_ids [массив идентификаторов id для открытых данных]
    #x_3413 -> closed_data [массив значений признаков (features) для закрытых данных]
    #y_3413 -> closed_target [массив target значений для закрытых данных] (это вычисляется в результате работы модели)
    #id_3413 -> closed_ids [массив идентификаторов id для закрытых данных]
    #

    #opened_ids:np.ndarray=np.ndarray([f'{i:04}'for i in range(opened_data.shape[0])]);
    opened_ids:np.ndarray=np.ndarray(shape=(opened_data.shape[0],),dtype='U4');
    for i in range(opened_data.shape[0]):
        opened_ids[i]=f'{i:04}';
        #print(f'i: {i}, opened_ids[i]: {opened_ids[i]}');
    print(f'opened_ids[0]: {opened_ids[0]}, opened_ids[799]: {opened_ids[799]}');
    print(f'opened_ids.shape: {opened_ids.shape}');

    closed_ids:np.ndarray=np.ndarray(shape=(closed_data.shape[0],),dtype='U4');
    for i in range(closed_data.shape[0]):
        closed_ids[i]=f'{i:04}';
        #print(f'i: {i}, opened_ids[i]: {opened_ids[i]}');
    print(f'closed_ids[0]: {closed_ids[0]}, closed_ids[199]: {closed_ids[199]}');
    print(f'closed_ids.shape: {closed_ids.shape}');
    #В программе для задачи F. Биометрия было так (13936 образцов открытых данных, 3413 образцов закрытых данных, 1536 признаков):
    #x.shape: (13936, 1536), y.shape: (13936,), ids.shape: (13936,),
    #x.shape: (3413, 1536), y.shape: (3413,), ids.shape: (3413,),

    return opened_data,opened_target,opened_ids,closed_data,closed_target,closed_ids;

def create_log_files()->None:
    """Функция создаёт log файлы (если они не существуют)"""
    if pathlib.Path('log.txt').exists()==False:#Создать файл log.txt если его не существует
        with open(file='log.txt',mode='wt',encoding='UTF-8')as f_log:
            pass;
    if pathlib.Path('log.csv').exists()==False:#Создать файл log.csv если его не существует и заполнить его заголовок
        with open(file='log.csv',mode='wt',encoding='UTF-8')as f_log:
            header_str:str=f'model_id,model_type,score_type,score_valid_mean,score_test';
            print(header_str,file=f_log);
            pass;
    pass;

def run_one_model_experiment_v5(problem_type:str,task_output:str,score_type:str,fbeta_score_beta:float=1.0,d2_pinball_score_alpha:float=0.5,d2_tweedie_score_power:float=0.0,mean_pinball_loss_alpha:float=0.5,mean_tweedie_deviance_power:float=0.0,model_type:str=None,model_hyperparams:dict=None,num_folds:int=10,score_valid_min_threshold:float=None,score_valid_max_threshold:float=None)->str:
    """
    Запуск одного эксперимента со случайным выбором модели и её гиперпараметров\n
    problem_type='classification'|'regression'\n
    task_output='mono_output'|'multi_output'\n
    score_type for classification (from sklearn.metrics import ...):\n
    accuracy_score,auc,average_precision_score,balanced_accuracy_score,brier_score_loss\n
    cohen_kappa_score,dcg_score,f1_score,fbeta_score,hamming_loss,hinge_loss,jaccard_score\n
    log_loss,matthews_corrcoef,ndcg_score,precision_score,recall_score,roc_auc_score,zero_one_loss\n
    score_type for regression (from sklearn.metrics import ...):\n
    d2_absolute_error_score,d2_pinball_score,d2_tweedie_score,explained_variance_score,max_error\n
    mean_absolute_percentage_error,mean_gamma_deviance,mean_pinball_loss,mean_poisson_deviance\n
    mean_squared_error,mean_squared_log_error,mean_tweedie_deviance,median_absolute_error\n
    r2_score,root_mean_squared_error,root_mean_squared_log_error\n
    """
    print(f'Функция run_one_model_experiment_v5 вызвана с параметрами: {locals()}');
    error_str:str='MODEL_WITH_ERROR';
    # 1. Загрузка ВСЕХ данных (открытых и закрытых)
    #Загрузка выполняется отдельно, так как:
    #1) Если эксперимент повторяется много раз, загружать данные каждый раз неэффективно по времени
    #2) Данные могут быть представлены в разных форматах (csv,json,npy,...)

    # 2. Разделение на train (для CV) и final test (только для оценки!)
    split_random_state:int=int(time.time()*(10**9))%(2**32);#Количество наносекунд с начала эпохи Unix -> [0, 4294967295]
    hyperparam_random_state:int=int(random.uniform(a=0.0,b=1e20))%(2**32);
    #print(f'split_random_state: {split_random_state}, hyperparam_random_state: {hyperparam_random_state}');

    opened_data_len:int=opened_data.shape[0];
    #print(f'opened_data.shape: {opened_data.shape}, opened_target.shape: {opened_target.shape}, opened_ids.shape: {opened_ids.shape}, ');
    #print(f'opened_data[0,0]: {opened_data[0,0]}, opened_target[0]: {opened_target[0]}, opened_ids[0]: {opened_ids[0]}, ');
    #x_13936.shape: (13936, 1536), y_13936.shape: (13936,), id_13936.shape: (13936,), 
    #x_13936[0,0]: 0.0009028149, y_13936[0]: 1, id_13936[0]: 5d1f7e43366513a1d0a6ec5640c3dc24,
    #Размер train для CV: (11150, 1536)
    #Размер final test: (2786, 1536)


    #split_random_state: 4248034816, hyperparam_random_state: 2880323584
    #opened_data.shape: (800, 7), opened_target.shape: (800,), opened_ids.shape: (800,), 
    #opened_data[0,0]: -0.5658143703611535, opened_target[0]: 0.3828701538424451, opened_ids[0]: 0000,
    #Возникло исключение <class 'Exception'>
    if problem_type=='classification':
        X_train_cv,X_test_final,y_train_cv,y_test_final,ids_train_cv,ids_test_final=train_test_split(
        opened_data,opened_target,opened_ids,test_size=0.2,random_state=split_random_state,stratify=opened_target);
    elif problem_type=='regression':
        X_train_cv,X_test_final,y_train_cv,y_test_final,ids_train_cv,ids_test_final=train_test_split(
        opened_data,opened_target,opened_ids,test_size=0.2,random_state=split_random_state);
    # Проверяем размеры
    #print(f"Размер train для CV: {X_train_cv.shape}")
    #print(f"Размер final test: {X_test_final.shape}")
    print(f'Доля X_train_cv от opened_data: {X_train_cv.shape[0]/opened_data_len}');#0.8
    print(f'Доля X_test_final от opened_data: {X_test_final.shape[0]/opened_data_len}');#0.2

    # 3. Установка типа и параметров модели
    if problem_type=='classification':
        if model_type is None:
            #Выбор случайной модели из списка:
            model_types:list[str]=['AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier','HistGradientBoostingClassifier','RandomForestClassifier','XGBClassifier','LGBMClassifier','LogisticRegression','PassiveAggressiveClassifier','Perceptron','RidgeClassifier','SGDClassifier'];
            model_type:str=random.choice(seq=model_types);
        if model_hyperparams is None:
            if model_type=='AdaBoostClassifier':model_hyperparams={'n_estimators':random.randint(a=20,b=500),'learning_rate':10**random.uniform(a=-2,b=0.5),'random_state':hyperparam_random_state};
            elif model_type=='BaggingClassifier':model_hyperparams={'n_estimators':random.randint(a=20,b=500),'max_samples':random.uniform(a=0.5,b=0.9),'max_features':random.uniform(a=0.5,b=0.9),'bootstrap':bool(random.randint(a=0,b=1)),'bootstrap_features':bool(random.randint(a=0,b=1)),'oob_score':bool(random.randint(a=0,b=1)),'warm_start':bool(random.randint(a=0,b=1)),'random_state':hyperparam_random_state,'n_jobs':-1};
            elif model_type=='ExtraTreesClassifier':model_hyperparams={'n_estimators':random.randint(a=20,b=500),'criterion':random.choice(seq=['gini','entropy','log_loss']),'max_depth':random.randint(a=5,b=50),'min_samples_split':random.randint(a=1,b=10),'min_samples_leaf':random.randint(a=1,b=10),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_features':random.choice(seq=['sqrt','log2',None]),'max_leaf_nodes':random.randint(a=5,b=50),'min_impurity_decrease':random.uniform(a=0.0,b=0.1),'bootstrap':bool(random.randint(a=0,b=1)),'oob_score':bool(random.randint(a=0,b=1)),'warm_start':bool(random.randint(a=0,b=1)),'class_weight':random.choice(seq=['balanced','balanced_subsample',None]),'ccp_alpha':random.uniform(a=0.0,b=0.1),'max_samples':random.uniform(a=0.001,b=1.0),'random_state':hyperparam_random_state,'n_jobs':-1};
            elif model_type=='GradientBoostingClassifier':model_hyperparams={'loss':random.choice(seq=['log_loss','exponential']),'learning_rate':10**random.uniform(a=-5,b=2),'n_estimators':random.randint(a=20,b=500),'subsample':random.uniform(a=0.001,b=1.0),'criterion':random.choice(seq=['friedman_mse','squared_error']),'min_samples_split':random.randint(a=2,b=10),'min_samples_leaf':random.randint(a=1,b=50),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_depth':random.randint(a=1,b=20),'min_impurity_decrease':random.uniform(a=0.0,b=1.0),'init':random.choice(seq=['zero',None]),'max_features':random.choice(seq=['sqrt','log2']),'max_leaf_nodes':random.randint(a=2,b=50),'warm_start':bool(random.randint(a=0,b=1)),'validation_fraction':random.uniform(a=0.00001,b=0.999999),'n_iter_no_change':random.randint(a=1,b=100),'tol':10**random.uniform(a=-8,b=-1),'ccp_alpha':random.uniform(a=0.0,b=0.1),'random_state':hyperparam_random_state};
            elif model_type=='HistGradientBoostingClassifier':model_hyperparams={'learning_rate':10**random.uniform(a=-4,b=0),'max_iter':random.randint(a=20,b=400),'max_leaf_nodes':random.randint(a=2,b=50),'max_depth':random.randint(a=2,b=40),'min_samples_leaf':random.randint(a=5,b=100),'l2_regularization':10**random.uniform(a=-10,b=0),'max_features':random.uniform(a=0.8,b=1.0),'max_bins':random.randint(a=10,b=255),'warm_start':random.choice(seq=[True,False]),'n_iter_no_change':random.randint(a=3,b=30),'tol':10**random.uniform(a=-10,b=-3),'random_state':hyperparam_random_state};
            elif model_type=='RandomForestClassifier':
                model_hyperparams={'n_estimators':random.randint(a=10,b=500),'criterion':random.choice(seq=['gini','entropy','log_loss']),'max_depth':random.randint(a=2,b=50),'min_samples_split':random.randint(a=2,b=50),'min_samples_leaf':random.randint(a=1,b=5),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.2),'max_features':random.choice(seq=['sqrt','log2',None]),'max_leaf_nodes':random.randint(a=10,b=100),'min_impurity_decrease':random.uniform(a=0.0,b=0.1),'bootstrap':random.choice(seq=[True,False]),'oob_score':random.choice(seq=[True,False]),'warm_start':random.choice(seq=[True,False]),'ccp_alpha':random.uniform(a=0.0,b=0.2),'max_samples':random.uniform(a=0.00000001,b=1.0),'random_state':hyperparam_random_state,'n_jobs':-1};
                #if model_hyperparams['bootstrap']==False:model_hyperparams['max_samples']=None;
            elif model_type=='XGBClassifier':model_hyperparams={'n_estimators':random.randint(a=10,b=500),'max_depth':random.randint(a=2,b=40),'max_leaves':random.randint(a=0,b=50),'max_bin':random.randint(a=5,b=100),'grow_policy':random.choice(seq=['depthwise','lossguide']),'learning_rate':10**random.uniform(a=-9,b=-1),'booster':random.choice(seq=['gbtree','gblinear','dart']),'gamma':random.uniform(a=0.0,b=1.0),'min_child_weight':random.uniform(a=0.01,b=0.1),'max_delta_step':random.uniform(a=0.1,b=2.0),'subsample':random.uniform(a=0.01,b=0.99),'sampling_method':random.choice(seq=['uniform','gradient_based']),'colsample_bytree':random.uniform(a=0.5,b=0.99),'colsample_bylevel':random.uniform(a=0.5,b=0.99),'colsample_bynode':random.uniform(a=0.5,b=0.99),'reg_alpha':10**random.uniform(a=-12,b=0),'reg_lambda':10**random.uniform(a=-12,b=0),'num_parallel_tree':random.randint(a=5,b=50),'importance_type':random.choice(seq=['gain','weight','cover','total_gain','total_cover']),'random_state':hyperparam_random_state,'n_jobs':-1};
            elif model_type=='LGBMClassifier':model_hyperparams={'num_leaves':random.randint(a=10,b=60),'max_depth':random.randint(4,40),'learning_rate':10**random.uniform(a=-4,b=1.5),'n_estimators':random.randint(a=20,b=500),'subsample_for_bin':random.randint(a=50_000,b=500_000),'min_child_weight':random.uniform(a=0.0001,b=0.01),'min_child_samples':random.randint(a=5,b=50),'reg_alpha':10**random.uniform(a=-12,b=0),'reg_lambda':10**random.uniform(a=-12,b=0),'random_state':hyperparam_random_state,'n_jobs':-1};
            elif model_type=='LogisticRegression':model_hyperparams={'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'dual':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-8,b=0),'C':random.uniform(a=0.5,b=1.5),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']),'max_iter':random.randint(a=50,b=500),'warm_start':random.choice(seq=[True,False]),'l1_ratio':random.uniform(a=0.0,b=1.0),'random_state':hyperparam_random_state,'n_jobs':-1};
            elif model_type=='PassiveAggressiveClassifier':model_hyperparams={'C':random.uniform(a=0.5,b=1.5),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=3000),'tol':10**random.uniform(a=-7,b=1),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'shuffle':random.choice(seq=[True,False]),'loss':random.choice(seq=['hinge','squared_hinge']),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[True,False,1,2,3,4,5,6,7,8,9,10]),'random_state':hyperparam_random_state,'n_jobs':-1};
            elif model_type=='Perceptron':model_hyperparams={'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-12,b=0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-6,b=0),'shuffle':random.choice(seq=[True,False]),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'warm_start':random.choice(seq=[True,False]),'random_state':hyperparam_random_state,'n_jobs':-1};
            elif model_type=='RidgeClassifier':model_hyperparams={'alpha':10**random.uniform(a=-2,b=1),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-8,b=0),'solver':random.choice(seq=['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']),'positive':random.choice(seq=[True,False]),'random_state':hyperparam_random_state,'n_jobs':-1};
            elif model_type=='SGDClassifier':model_hyperparams={'loss':random.choice(seq=['hinge','log_loss','modified_huber','squared_hinge','perceptron','squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']),'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-8,b=0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-6,b=0),'shuffle':random.choice(seq=[True,False]),'epsilon':10**random.uniform(a=-9,b=3),'learning_rate':random.choice(seq=['constant','optimal','invscaling','adaptive']),'eta0':10**random.uniform(a=-5,b=1),'power_t':random.uniform(a=-5,b=6),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[True,False,1,2,3,4,5,6,7,8,9,10]),'random_state':hyperparam_random_state,'n_jobs':-1};
    elif problem_type=='regression':
        if model_type is None:
            #Выбор случайной модели из списка:
            if task_output=='mono_output':
                model_types:list[str]=['LinearRegression','Ridge','SGDRegressor','ElasticNet','Lars','Lasso','LassoLars','LassoLarsIC','ARDRegression','BayesianRidge','HuberRegressor','QuantileRegressor','RANSACRegressor','TheilSenRegressor','GammaRegressor','PoissonRegressor','TweedieRegressor','PassiveAggressiveRegressor','AdaBoostRegressor','BaggingRegressor','ExtraTreesRegressor','GradientBoostingRegressor','HistGradientBoostingRegressor','RandomForestRegressor'];
            elif task_output=='multi_output':
                model_types:list[str]=['LinearRegression','Ridge','SGDRegressor','ElasticNet','Lars','Lasso','LassoLars','LassoLarsIC','ARDRegression','BayesianRidge','MultiTaskElasticNet','MultiTaskLasso','HuberRegressor','QuantileRegressor','RANSACRegressor','TheilSenRegressor','GammaRegressor','PoissonRegressor','TweedieRegressor','PassiveAggressiveRegressor','AdaBoostRegressor','BaggingRegressor','ExtraTreesRegressor','GradientBoostingRegressor','HistGradientBoostingRegressor','RandomForestRegressor'];
            else:
                print(f'Необходимо задать тип выхода (параметр task_output:str, значения: mono_output или multi_output)');
                return error_str;
            model_type:str=random.choice(seq=model_types);
        if model_hyperparams is None:
            if model_type=='LinearRegression':model_hyperparams={'fit_intercept':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-9,b=-3),'positive':random.choice(seq=[True,False])};
            elif model_type=='Ridge':model_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=500,b=20000),'tol':10**random.uniform(a=-9,b=-0.1),'solver':random.choice(seq=['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']),'positive':bool(random.randint(a=0,b=1)),'random_state':hyperparam_random_state};
            elif model_type=='SGDRegressor':model_hyperparams={'loss':random.choice(seq=['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']),'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-9,b=1.0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=20000),'tol':10**random.uniform(a=-6,b=0),'epsilon':10**random.uniform(a=-3,b=1),'learning_rate':random.choice(seq=['constant','optimal','invscaling','adaptive']),'eta0':10**random.uniform(a=-5,b=0),'power_t':random.uniform(a=-100,b=100),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0,b=1),'n_iter_no_change':random.randint(a=2,b=10),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[False,False,False,False,False,False,False,False,False,False,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),'random_state':hyperparam_random_state};
            elif model_type=='ElasticNet':model_hyperparams={'alpha':10**random.uniform(a=-5,b=2),'l1_ratio':random.uniform(a=0,b=1),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-7,b=-1),'warm_start':random.choice(seq=[True,False]),'positive':random.choice(seq=[True,False]),'selection':random.choice(seq=['cyclic','random']),'random_state':hyperparam_random_state};
            elif model_type=='Lars':model_hyperparams={'fit_intercept':random.choice(seq=[True,False]),'n_nonzero_coefs':random.randint(a=10,b=100),'eps':10**random.uniform(a=-5,b=-1),'fit_path':random.choice(seq=[True,False]),'jitter':10**random.uniform(a=-9,b=-1),'random_state':hyperparam_random_state};
            elif model_type=='Lasso':model_hyperparams={'alpha':10**random.uniform(a=-5,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(100,2000),'tol':10**random.uniform(a=-8,b=-1),'warm_start':random.choice(seq=[True,False]),'positive':random.choice(seq=[True,False]),'selection':random.choice(seq=['cyclic','random']),'random_state':hyperparam_random_state};
            elif model_type=='LassoLars':model_hyperparams={'alpha':10**random.uniform(a=-5,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=1500),'eps':10**random.uniform(a=-10,b=-5),'fit_path':random.choice(seq=[True,False]),'positive':random.choice(seq=[True,False]),'jitter':10**random.uniform(a=-9,b=-1),'random_state':hyperparam_random_state};
            elif model_type=='LassoLarsIC':model_hyperparams={'criterion':random.choice(seq=['aic','bic']),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=1500),'eps':10**random.uniform(a=-16,b=-10),'positive':random.choice(seq=[True,False]),'noise_variance':10**random.uniform(a=-5,b=-1)};
            elif model_type=='ARDRegression':model_hyperparams={'max_iter':random.randint(a=100,b=700),'tol':10**random.uniform(a=-7,b=-1),'alpha_1':10**random.uniform(a=-10,b=-2),'alpha_2':10**random.uniform(a=-10,b=-2),'lambda_1':10**random.uniform(a=-10,b=-2),'lambda_2':10**random.uniform(a=-10,b=-2),'compute_score':random.choice(seq=[True,False]),'threshold_lambda':random.uniform(a=5000,b=15000),'fit_intercept':random.choice(seq=[True,False])};
            elif model_type=='BayesianRidge':model_hyperparams={'max_iter':random.randint(a=100,b=700),'tol':10**random.uniform(a=-5,b=-1),'alpha_1':10**random.uniform(a=-10,b=-2),'alpha_2':10**random.uniform(a=-10,b=-2),'lambda_1':10**random.uniform(a=-10,b=-2),'lambda_2':10**random.uniform(a=-10,b=-2),'alpha_init':random.uniform(a=0.01,b=1.0),'lambda_init':random.uniform(a=0.01,b=1.0),'compute_score':random.choice(seq=[True,False]),'fit_intercept':random.choice(seq=[True,False])};
            elif model_type=='MultiTaskElasticNet':model_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=2000),'tol':10**random.uniform(a=-8,b=-1),'warm_start':random.choice(seq=[True,False]),'selection':random.choice(seq=['cyclic','random']),'random_state':hyperparam_random_state};
            elif model_type=='MultiTaskLasso':model_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=2000),'tol':10**random.uniform(a=-8,b=-1),'warm_start':random.choice(seq=[True,False]),'selection':random.choice(seq=['cyclic','random']),'random_state':hyperparam_random_state};
            elif model_type=='HuberRegressor':model_hyperparams={'epsilon':random.uniform(a=0.0,b=10.0),'max_iter':random.randint(a=20,b=200),'alpha':10**random.uniform(a=-8,b=0),'warm_start':random.choice(seq=[True,False]),'fit_intercept':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-8,b=-2)};
            elif model_type=='QuantileRegressor':model_hyperparams={'quantile':random.uniform(a=0.0,b=1.0),'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['highs-ds','highs-ipm','highs','interior-point','revised simplex'])};
            elif model_type=='RANSACRegressor':model_hyperparams={'min_samples':random.uniform(a=0.0,b=1.0),'max_trials':random.randint(a=50,b=150),'max_skips':random.randint(a=500,b=1000),'stop_n_inliers':random.randint(a=500,b=1000),'stop_score':10**random.uniform(a=3,b=10),'stop_probability':random.uniform(a=0.95,b=1.00),'loss':random.choice(seq=['absolute_error','squared_error']),'random_state':hyperparam_random_state};
            elif model_type=='TheilSenRegressor':model_hyperparams={'fit_intercept':random.choice(seq=[True,False]),'max_subpopulation':10**random.uniform(a=-6,b=-2),'n_subsamples':random.randint(a=opened_data.shape[1]+1,b=opened_data.shape[0]),'max_iter':random.randint(a=100,b=500),'tol':10**random.uniform(a=-5,b=-1),'n_jobs':-1,'random_state':hyperparam_random_state};
            elif model_type=='GammaRegressor':model_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-9,b=-2),'warm_start':random.choice(seq=[True,False])};
            elif model_type=='PoissonRegressor':model_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-6,b=-2),'warm_start':random.choice(seq=[True,False])};
            elif model_type=='TweedieRegressor':model_hyperparams={'power':random.choice(seq=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.01,1.02,1.03,1.04,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,1.96,1.97,1.98,1.99,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0]),'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'link':random.choice(seq=['auto','identity','log']),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-6,b=-2),'warm_start':random.choice(seq=[True,False])};
            elif model_type=='PassiveAggressiveRegressor':model_hyperparams={'C':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=2000),'tol':10**random.uniform(a=-5,b=-1),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=0.9),'n_iter_no_change':random.randint(a=2,b=10),'loss':random.choice(seq=['epsilon_insensitive','squared_epsilon_insensitive']),'epsilon':random.uniform(a=0.05,b=0.15),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[False,False,False,False,False,False,False,False,False,False,False,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]),'random_state':hyperparam_random_state};
            elif model_type=='AdaBoostRegressor':model_hyperparams={'n_estimators':random.randint(a=10,b=500),'learning_rate':10**random.uniform(a=-3,b=2),'loss':random.choice(seq=['linear','square','exponential']),'random_state':hyperparam_random_state};
            elif model_type=='BaggingRegressor':model_hyperparams={'n_estimators':random.randint(a=5,b=30),'max_samples':random.uniform(a=0.2,b=1.0),'max_features':random.uniform(a=0.2,b=1.0),'bootstrap':random.choice(seq=[True,False]),'bootstrap_features':random.choice(seq=[True,False]),'oob_score':random.choice(seq=[True,False]),'warm_start':random.choice(seq=[True,False]),'n_jobs':-1,'random_state':hyperparam_random_state};
            elif model_type=='ExtraTreesRegressor':model_hyperparams={'n_estimators':random.randint(a=20,b=200),'criterion':random.choice(seq=['squared_error','absolute_error','friedman_mse','poisson']),'max_depth':random.randint(a=5,b=20),'min_samples_split':random.uniform(a=0.0,b=1.0),'min_samples_leaf':random.uniform(a=0.0,b=1.0),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_features':random.choice(seq=['sqrt','log2',None]),'max_leaf_nodes':random.randint(a=5,b=10),'min_impurity_decrease':random.uniform(a=0.0,b=0.5),'bootstrap':random.choice(seq=[True,False]),'oob_score':random.choice(seq=[True,False]),'n_jobs':-1,'warm_start':random.choice(seq=[True,False]),'ccp_alpha':random.uniform(a=0.0,b=0.1),'max_samples':random.uniform(a=0.0,b=1.0),'random_state':hyperparam_random_state};
            elif model_type=='GradientBoostingRegressor':model_hyperparams={'loss':random.choice(seq=['squared_error','absolute_error','huber','quantile']),'learning_rate':10**random.uniform(a=-2,b=0),'n_estimators':random.randint(a=20,b=300),'subsample':random.uniform(a=0.0,b=1.0),'criterion':random.choice(seq=['friedman_mse','squared_error']),'min_samples_split':random.uniform(a=0.0,b=1.0),'min_samples_leaf':random.uniform(a=0.0,b=1.0),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_depth':random.randint(a=1,b=7),'min_impurity_decrease':random.uniform(a=0.0,b=1.0),'max_features':random.choice(seq=['sqrt','log2','sqrt','log2','sqrt','log2','sqrt','log2',0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,None,None,None,None,None,None]),'alpha':random.uniform(a=0.0,b=1.0),'max_leaf_nodes':random.randint(a=2,b=100),'warm_start':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=0.4),'n_iter_no_change':random.randint(a=5,b=20),'tol':10**random.uniform(a=-6,b=-2),'ccp_alpha':random.uniform(a=0.0,b=100.0),'random_state':hyperparam_random_state};
            elif model_type=='HistGradientBoostingRegressor':model_hyperparams={'loss':random.choice(seq=['squared_error','absolute_error','gamma','poisson','quantile']),'quantile':random.uniform(a=0.0,b=1.0),'learning_rate':10**random.uniform(a=-2,b=0),'max_iter':random.randint(a=20,b=200),'max_leaf_nodes':random.randint(a=2,b=60),'max_depth':random.randint(a=2,b=10),'min_samples_leaf':random.randint(a=5,b=50),'l2_regularization':random.uniform(a=0.0,b=1.0),'max_features':random.uniform(a=0.2,b=1.0),'max_bins':random.randint(a=10,b=255),'warm_start':random.choice(seq=[True,False]),'early_stopping':random.choice(seq=['auto',True]),'scoring':random.choice(seq=['loss',None]),'validation_fraction':random.uniform(a=0.05,b=0.25),'n_iter_no_change':random.randint(a=3,b=30),'tol':10**random.uniform(a=-11,b=-3),'random_state':hyperparam_random_state};
            elif model_type=='RandomForestRegressor':model_hyperparams={'n_estimators':random.randint(a=20,b=200),'criterion':random.choice(seq=['squared_error','absolute_error','friedman_mse','poisson']),'max_depth':random.randint(a=2,b=20),'min_samples_split':random.uniform(a=0.0,b=1.0),'min_samples_leaf':random.uniform(a=0.0,b=1.0),'min_weight_fraction_leaf':random.uniform(a=0.0,b=1.0),'max_features':random.choice(seq=['sqrt','log2','sqrt','log2','sqrt','log2','sqrt','log2','sqrt','log2',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),'min_impurity_decrease':random.uniform(a=0.0,b=0.2),'bootstrap':random.choice(seq=[True,False]),'warm_start':random.choice(seq=[True,False]),'ccp_alpha':random.uniform(a=0.0,b=1.0),'max_samples':random.uniform(a=0.0,b=1.0),'n_jobs':-1,'random_state':hyperparam_random_state};
            
    # 4. Подготовка кросс-валидации
    if problem_type=='classification':K_Fold:StratifiedKFold=StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=split_random_state)
    elif problem_type=='regression':K_Fold:KFold=KFold(n_splits=num_folds,shuffle=True,random_state=split_random_state);
    #StratifiedKFold предназначен ТОЛЬКО для классификационных задач, где целевая переменная имеет дискретные значения (бинарные или 
    #мультикласс). В регрессии же целевая переменная непрерывная (continuous), и StratifiedKFold не может работать с такими данными.
    
    #scaler:StandardScaler=StandardScaler();
    valid_scores=[];

    print(f"Начинаем кросс-валидацию модели {model_type} с гиперпараметрами: {model_hyperparams}...")
    for fold_num,(train_index,valid_index) in enumerate(K_Fold.split(X=X_train_cv,y=y_train_cv)):
        print(f"  Обрабатываем фолд {fold_num+1}/{num_folds}...",end=' ');

        # 4.1 Разделение на train/valid для фолда
        X_train_fold, X_valid_fold = X_train_cv[train_index], X_train_cv[valid_index]
        y_train_fold, y_valid_fold = y_train_cv[train_index], y_train_cv[valid_index]
        print(f'Доля X_train_fold от opened_data: {X_train_fold.shape[0]/opened_data_len}');#0.72
        print(f'Доля X_valid_fold от opened_data: {X_valid_fold.shape[0]/opened_data_len}');#0.08

        # 4.2 & 4.3 Масштабирование
        scaler_cross_valid:StandardScaler=StandardScaler();#Старый scaler (если он был в переменной) перезаписывается новым экземпляром класса
        scaler_cross_valid.fit(X_train_fold);#fit только на train, без valid
        X_train_fold_scaled=scaler_cross_valid.transform(X_train_fold)
        X_valid_fold_scaled=scaler_cross_valid.transform(X_valid_fold)

        # 4.4 & 4.5 Обучение и оценка
        if problem_type=='classification':
            if model_type=='AdaBoostClassifier':model_cross_valid=AdaBoostClassifier(**model_hyperparams);
            elif model_type=='BaggingClassifier':model_cross_valid=BaggingClassifier(**model_hyperparams);
            elif model_type=='ExtraTreesClassifier':model_cross_valid=ExtraTreesClassifier(**model_hyperparams);
            elif model_type=='GradientBoostingClassifier':model_cross_valid=GradientBoostingClassifier(**model_hyperparams);
            elif model_type=='HistGradientBoostingClassifier':model_cross_valid=HistGradientBoostingClassifier(**model_hyperparams);
            elif model_type=='RandomForestClassifier':model_cross_valid=RandomForestClassifier(**model_hyperparams);
            elif model_type=='XGBClassifier':model_cross_valid=XGBClassifier(**model_hyperparams);
            elif model_type=='LGBMClassifier':model_cross_valid=LGBMClassifier(**model_hyperparams);
            elif model_type=='LogisticRegression':model_cross_valid=LogisticRegression(**model_hyperparams);
            elif model_type=='PassiveAggressiveClassifier':model_cross_valid=PassiveAggressiveClassifier(**model_hyperparams);
            elif model_type=='Perceptron':model_cross_valid=Perceptron(**model_hyperparams);
            elif model_type=='RidgeClassifier':model_cross_valid=RidgeClassifier(**model_hyperparams);
            elif model_type=='SGDClassifier':model_cross_valid=SGDClassifier(**model_hyperparams);
        elif problem_type=='regression':
            if model_type=='LinearRegression':model_cross_valid=LinearRegression(**model_hyperparams);
            elif model_type=='Ridge':model_cross_valid=Ridge(**model_hyperparams);
            elif model_type=='SGDRegressor':model_cross_valid=SGDRegressor(**model_hyperparams);
            elif model_type=='ElasticNet':model_cross_valid=ElasticNet(**model_hyperparams);
            elif model_type=='Lars':model_cross_valid=Lars(**model_hyperparams);
            elif model_type=='Lasso':model_cross_valid=Lasso(**model_hyperparams);
            elif model_type=='LassoLars':model_cross_valid=LassoLars(**model_hyperparams);
            elif model_type=='LassoLarsIC':model_cross_valid=LassoLarsIC(**model_hyperparams);
            elif model_type=='ARDRegression':model_cross_valid=ARDRegression(**model_hyperparams);
            elif model_type=='BayesianRidge':model_cross_valid=BayesianRidge(**model_hyperparams);
            elif model_type=='MultiTaskElasticNet':model_cross_valid=MultiTaskElasticNet(**model_hyperparams);
            elif model_type=='MultiTaskLasso':model_cross_valid=MultiTaskLasso(**model_hyperparams);
            elif model_type=='HuberRegressor':model_cross_valid=HuberRegressor(**model_hyperparams);
            elif model_type=='QuantileRegressor':model_cross_valid=QuantileRegressor(**model_hyperparams);
            elif model_type=='RANSACRegressor':model_cross_valid=RANSACRegressor(**model_hyperparams);
            elif model_type=='TheilSenRegressor':model_cross_valid=TheilSenRegressor(**model_hyperparams);
            elif model_type=='GammaRegressor':model_cross_valid=GammaRegressor(**model_hyperparams);
            elif model_type=='PoissonRegressor':model_cross_valid=PoissonRegressor(**model_hyperparams);
            elif model_type=='TweedieRegressor':model_cross_valid=TweedieRegressor(**model_hyperparams);
            elif model_type=='PassiveAggressiveRegressor':model_cross_valid=PassiveAggressiveRegressor(**model_hyperparams);
            elif model_type=='AdaBoostRegressor':model_cross_valid=AdaBoostRegressor(**model_hyperparams);
            elif model_type=='BaggingRegressor':model_cross_valid=BaggingRegressor(**model_hyperparams);
            elif model_type=='ExtraTreesRegressor':model_cross_valid=ExtraTreesRegressor(**model_hyperparams);
            elif model_type=='GradientBoostingRegressor':model_cross_valid=GradientBoostingRegressor(**model_hyperparams);
            elif model_type=='HistGradientBoostingRegressor':model_cross_valid=HistGradientBoostingRegressor(**model_hyperparams);
            elif model_type=='RandomForestRegressor':model_cross_valid=RandomForestRegressor(**model_hyperparams);
            
        model_cross_valid.fit(X_train_fold_scaled, y_train_fold);

        y_valid_pred=model_cross_valid.predict(X_valid_fold_scaled);
        if problem_type=='classification':
            if score_type=='accuracy_score':score_valid=accuracy_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='auc':score_valid=auc(x=y_valid_fold,y=y_valid_pred);
            elif score_type=='average_precision_score':pass;
            elif score_type=='balanced_accuracy_score':score_valid=balanced_accuracy_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='brier_score_loss':score_valid=brier_score_loss(y_true=y_valid_fold,y_prob=y_valid_pred);
            elif score_type=='cohen_kappa_score':score_valid=cohen_kappa_score(y1=y_valid_fold,y2=y_valid_pred);
            elif score_type=='dcg_score':pass;
            elif score_type=='f1_score':score_valid=f1_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='fbeta_score':score_valid=fbeta_score(y_true=y_valid_fold,y_pred=y_valid_pred,beta=fbeta_score_beta);
            elif score_type=='hamming_loss':score_valid=hamming_loss(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='hinge_loss':pass;
            elif score_type=='jaccard_score':score_valid=jaccard_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='log_loss':score_valid=log_loss(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='matthews_corrcoef':score_valid=matthews_corrcoef(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='ndcg_score':pass;
            elif score_type=='precision_score':score_valid=precision_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='recall_score':score_valid=recall_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='roc_auc_score':pass;
            elif score_type=='zero_one_loss':score_valid=zero_one_loss(y_true=y_valid_fold,y_pred=y_valid_pred);
        elif problem_type=='regression':
            if score_type=='d2_absolute_error_score':score_valid=d2_absolute_error_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='d2_pinball_score':score_valid=d2_pinball_score(y_true=y_valid_fold,y_pred=y_valid_pred,alpha=d2_pinball_score_alpha);
            elif score_type=='d2_tweedie_score':score_valid=d2_tweedie_score(y_true=y_valid_fold,y_pred=y_valid_pred,power=d2_tweedie_score_power);
            elif score_type=='explained_variance_score':score_valid=explained_variance_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='max_error':score_valid=max_error(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='mean_absolute_percentage_error':score_valid=mean_absolute_percentage_error(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='mean_gamma_deviance':score_valid=mean_gamma_deviance(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='mean_pinball_loss':score_valid=mean_pinball_loss(y_true=y_valid_fold,y_pred=y_valid_pred,alpha=mean_pinball_loss_alpha);
            elif score_type=='mean_poisson_deviance':score_valid=mean_poisson_deviance(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='mean_squared_error':score_valid=mean_squared_error(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='mean_squared_log_error':score_valid=mean_squared_log_error(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='mean_tweedie_deviance':score_valid=mean_tweedie_deviance(y_true=y_valid_fold,y_pred=y_valid_pred,power=mean_tweedie_deviance_power);
            elif score_type=='median_absolute_error':score_valid=median_absolute_error(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='r2_score':score_valid=r2_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='root_mean_squared_error':score_valid=root_mean_squared_error(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='root_mean_squared_log_error':score_valid=root_mean_squared_log_error(y_true=y_valid_fold,y_pred=y_valid_pred);
        print(f"Тип метрики: {score_type}, значение метрики на фолде {fold_num+1} из {num_folds}: {score_valid:.6f}");
        if score_valid_min_threshold!=None:
            if score_valid<score_valid_min_threshold:
                print(f'Значение метрики ниже установленного минимального порога ({score_valid_min_threshold:.6f}), валидация этой модели с этим набором гиперпараметров прервана для экономии времени');
                return error_str;
        elif score_valid_max_threshold!=None:
            if score_valid>score_valid_max_threshold:
                print(f'Значение метрики выше установленного максимального порога ({score_valid_max_threshold:.6f}), валидация этой модели с этим набором гиперпараметров прервана для экономии времени');
                return error_str;
        valid_scores.append(score_valid);
        

    # 5. Расчет средней точности по CV
    score_valid_mean=np.mean(valid_scores);
    print(f"\nСреднее значение метрики {score_type} по кросс-валидации: {score_valid_mean:.6f}");

    # 6. Оценка качества на отложенной выборке (20% открытых данных)
    #print("Оценка качества на отложенной выборке (20% открытых данных)...")
    scaler_for_final_test:StandardScaler=StandardScaler();
    X_train_cv_scaled=scaler_for_final_test.fit_transform(X_train_cv)#scaler учится на 80% открытых данных (то есть на всех данных, которые испоьзовались для кросс-валидации)
    X_test_final_scaled=scaler_for_final_test.transform(X_test_final)#применяется к 20% открытых данных (то есть на тех данных, которые НЕ использовались для кросс-валидации)
    if problem_type=='classification':
        if model_type=='AdaBoostClassifier':model_for_final_test=AdaBoostClassifier(**model_hyperparams);
        elif model_type=='BaggingClassifier':model_for_final_test=BaggingClassifier(**model_hyperparams);
        elif model_type=='ExtraTreesClassifier':model_for_final_test=ExtraTreesClassifier(**model_hyperparams);
        elif model_type=='GradientBoostingClassifier':model_for_final_test=GradientBoostingClassifier(**model_hyperparams);
        elif model_type=='HistGradientBoostingClassifier':model_for_final_test=HistGradientBoostingClassifier(**model_hyperparams);
        elif model_type=='RandomForestClassifier':model_for_final_test=RandomForestClassifier(**model_hyperparams);
        elif model_type=='XGBClassifier':model_for_final_test=XGBClassifier(**model_hyperparams);
        elif model_type=='LGBMClassifier':model_for_final_test=LGBMClassifier(**model_hyperparams);
        elif model_type=='LogisticRegression':model_for_final_test=LogisticRegression(**model_hyperparams);
        elif model_type=='PassiveAggressiveClassifier':model_for_final_test=PassiveAggressiveClassifier(**model_hyperparams);
        elif model_type=='Perceptron':model_for_final_test=Perceptron(**model_hyperparams);
        elif model_type=='RidgeClassifier':model_for_final_test=RidgeClassifier(**model_hyperparams);
        elif model_type=='SGDClassifier':model_for_final_test=SGDClassifier(**model_hyperparams);
    elif problem_type=='regression':
        if model_type=='LinearRegression':model_for_final_test=LinearRegression(**model_hyperparams);
        elif model_type=='Ridge':model_for_final_test=Ridge(**model_hyperparams);
        elif model_type=='SGDRegressor':model_for_final_test=SGDRegressor(**model_hyperparams);
        elif model_type=='ElasticNet':model_for_final_test=ElasticNet(**model_hyperparams);
        elif model_type=='Lars':model_for_final_test=Lars(**model_hyperparams);
        elif model_type=='Lasso':model_for_final_test=Lasso(**model_hyperparams);
        elif model_type=='LassoLars':model_for_final_test=LassoLars(**model_hyperparams);
        elif model_type=='LassoLarsIC':model_for_final_test=LassoLarsIC(**model_hyperparams);
        elif model_type=='ARDRegression':model_for_final_test=ARDRegression(**model_hyperparams);
        elif model_type=='BayesianRidge':model_for_final_test=BayesianRidge(**model_hyperparams);
        elif model_type=='MultiTaskElasticNet':model_for_final_test=MultiTaskElasticNet(**model_hyperparams);
        elif model_type=='MultiTaskLasso':model_for_final_test=MultiTaskLasso(**model_hyperparams);
        elif model_type=='HuberRegressor':model_for_final_test=HuberRegressor(**model_hyperparams);
        elif model_type=='QuantileRegressor':model_for_final_test=QuantileRegressor(**model_hyperparams);
        elif model_type=='RANSACRegressor':model_for_final_test=RANSACRegressor(**model_hyperparams);
        elif model_type=='TheilSenRegressor':model_for_final_test=TheilSenRegressor(**model_hyperparams);
        elif model_type=='GammaRegressor':model_for_final_test=GammaRegressor(**model_hyperparams);
        elif model_type=='PoissonRegressor':model_for_final_test=PoissonRegressor(**model_hyperparams);
        elif model_type=='TweedieRegressor':model_for_final_test=TweedieRegressor(**model_hyperparams);
        elif model_type=='PassiveAggressiveRegressor':model_for_final_test=PassiveAggressiveRegressor(**model_hyperparams);
        elif model_type=='AdaBoostRegressor':model_for_final_test=AdaBoostRegressor(**model_hyperparams);
        elif model_type=='BaggingRegressor':model_for_final_test=BaggingRegressor(**model_hyperparams);
        elif model_type=='ExtraTreesRegressor':model_for_final_test=ExtraTreesRegressor(**model_hyperparams);
        elif model_type=='GradientBoostingRegressor':model_for_final_test=GradientBoostingRegressor(**model_hyperparams);
        elif model_type=='HistGradientBoostingRegressor':model_for_final_test=HistGradientBoostingRegressor(**model_hyperparams);
        elif model_type=='RandomForestRegressor':model_for_final_test=RandomForestRegressor(**model_hyperparams);


    model_for_final_test.fit(X_train_cv_scaled, y_train_cv);#Обучение на 80% открытых данных

    y_test_pred = model_for_final_test.predict(X_test_final_scaled);#Тестирование на 20% открытых данных
    if problem_type=='classification':
        if score_type=='accuracy_score':score_test=accuracy_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='auc':score_test=auc(x=y_test_final,y=y_test_pred);
        elif score_type=='average_precision_score':pass;
        elif score_type=='balanced_accuracy_score':score_test=balanced_accuracy_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='brier_score_loss':score_test=brier_score_loss(y_true=y_test_final,y_prob=y_test_pred);
        elif score_type=='cohen_kappa_score':score_test=cohen_kappa_score(y1=y_test_final,y2=y_test_pred);
        elif score_type=='dcg_score':pass;
        elif score_type=='f1_score':score_test=f1_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='fbeta_score':score_test=fbeta_score(y_true=y_test_final,y_pred=y_test_pred,beta=fbeta_score_beta);
        elif score_type=='hamming_loss':score_test=hamming_loss(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='hinge_loss':pass;
        elif score_type=='jaccard_score':score_test=jaccard_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='log_loss':score_test=log_loss(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='matthews_corrcoef':score_test=matthews_corrcoef(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='ndcg_score':pass;
        elif score_type=='precision_score':score_test=precision_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='recall_score':score_test=recall_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='roc_auc_score':pass;
        elif score_type=='zero_one_loss':score_test=zero_one_loss(y_true=y_test_final,y_pred=y_test_pred);
    elif problem_type=='regression':
        if score_type=='d2_absolute_error_score':score_test=d2_absolute_error_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='d2_pinball_score':score_test=d2_pinball_score(y_true=y_test_final,y_pred=y_test_pred,alpha=d2_pinball_score_alpha);
        elif score_type=='d2_tweedie_score':score_test=d2_tweedie_score(y_true=y_test_final,y_pred=y_test_pred,power=d2_tweedie_score_power);
        elif score_type=='explained_variance_score':score_test=explained_variance_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='max_error':score_test=max_error(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='mean_absolute_percentage_error':score_test=mean_absolute_percentage_error(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='mean_gamma_deviance':score_test=mean_gamma_deviance(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='mean_pinball_loss':score_test=mean_pinball_loss(y_true=y_test_final,y_pred=y_test_pred,alpha=mean_pinball_loss_alpha);
        elif score_type=='mean_poisson_deviance':score_test=mean_poisson_deviance(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='mean_squared_error':score_test=mean_squared_error(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='mean_squared_log_error':score_test=mean_squared_log_error(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='mean_tweedie_deviance':score_test=mean_tweedie_deviance(y_true=y_test_final,y_pred=y_test_pred,power=mean_tweedie_deviance_power);
        elif score_type=='median_absolute_error':score_test=median_absolute_error(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='r2_score':score_test=r2_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='root_mean_squared_error':score_test=root_mean_squared_error(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='root_mean_squared_log_error':score_test=root_mean_squared_log_error(y_true=y_test_final,y_pred=y_test_pred);
    print(f"Тип метрики: {score_type}, значение метрики на тесте (отложенная выборка, 20% открытых данных): {score_test:.6f}");
    
    #7. ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ НА 100% ОТКРЫТЫХ ДАННЫХ И ЕЁ СОХРАНЕНИЕ В *.pkl файл
    print("Обучение финальной production-модели на ВСЕХ данных...")
    #Создаем новый scaler, который учится на ВСЕХ образцах из opened_data (то есть из тех данных, для которых есть значения target)
    scaler_production:StandardScaler=StandardScaler();
    X_all_scaled=scaler_production.fit_transform(opened_data) #Важно: fit_transform для всех данных!
    if problem_type=='classification':
        if model_type=='AdaBoostClassifier':model_production=AdaBoostClassifier(**model_hyperparams);
        elif model_type=='BaggingClassifier':model_production=BaggingClassifier(**model_hyperparams);
        elif model_type=='ExtraTreesClassifier':model_production=ExtraTreesClassifier(**model_hyperparams);
        elif model_type=='GradientBoostingClassifier':model_production=GradientBoostingClassifier(**model_hyperparams);
        elif model_type=='HistGradientBoostingClassifier':model_production=HistGradientBoostingClassifier(**model_hyperparams);
        elif model_type=='RandomForestClassifier':model_production=RandomForestClassifier(**model_hyperparams);
        elif model_type=='XGBClassifier':model_production=XGBClassifier(**model_hyperparams);
        elif model_type=='LGBMClassifier':model_production=LGBMClassifier(**model_hyperparams);
        elif model_type=='LogisticRegression':model_production=LogisticRegression(**model_hyperparams);
        elif model_type=='PassiveAggressiveClassifier':model_production=PassiveAggressiveClassifier(**model_hyperparams);
        elif model_type=='Perceptron':model_production=Perceptron(**model_hyperparams);
        elif model_type=='RidgeClassifier':model_production=RidgeClassifier(**model_hyperparams);
        elif model_type=='SGDClassifier':model_production=SGDClassifier(**model_hyperparams);
    elif problem_type=='regression':
        if model_type=='LinearRegression':model_production=LinearRegression(**model_hyperparams);
        elif model_type=='Ridge':model_production=Ridge(**model_hyperparams);
        elif model_type=='SGDRegressor':model_production=SGDRegressor(**model_hyperparams);
        elif model_type=='ElasticNet':model_production=ElasticNet(**model_hyperparams);
        elif model_type=='Lars':model_production=Lars(**model_hyperparams);
        elif model_type=='Lasso':model_production=Lasso(**model_hyperparams);
        elif model_type=='LassoLars':model_production=LassoLars(**model_hyperparams);
        elif model_type=='LassoLarsIC':model_production=LassoLarsIC(**model_hyperparams);
        elif model_type=='ARDRegression':model_production=ARDRegression(**model_hyperparams);
        elif model_type=='BayesianRidge':model_production=BayesianRidge(**model_hyperparams);
        elif model_type=='MultiTaskElasticNet':model_production=MultiTaskElasticNet(**model_hyperparams);
        elif model_type=='MultiTaskLasso':model_production=MultiTaskLasso(**model_hyperparams);
        elif model_type=='HuberRegressor':model_production=HuberRegressor(**model_hyperparams);
        elif model_type=='QuantileRegressor':model_production=QuantileRegressor(**model_hyperparams);
        elif model_type=='RANSACRegressor':model_production=RANSACRegressor(**model_hyperparams);
        elif model_type=='TheilSenRegressor':model_production=TheilSenRegressor(**model_hyperparams);
        elif model_type=='GammaRegressor':model_production=GammaRegressor(**model_hyperparams);
        elif model_type=='PoissonRegressor':model_production=PoissonRegressor(**model_hyperparams);
        elif model_type=='TweedieRegressor':model_production=TweedieRegressor(**model_hyperparams);
        elif model_type=='PassiveAggressiveRegressor':model_production=PassiveAggressiveRegressor(**model_hyperparams);
        elif model_type=='AdaBoostRegressor':model_production=AdaBoostRegressor(**model_hyperparams);
        elif model_type=='BaggingRegressor':model_production=BaggingRegressor(**model_hyperparams);
        elif model_type=='ExtraTreesRegressor':model_production=ExtraTreesRegressor(**model_hyperparams);
        elif model_type=='GradientBoostingRegressor':model_production=GradientBoostingRegressor(**model_hyperparams);
        elif model_type=='HistGradientBoostingRegressor':model_production=HistGradientBoostingRegressor(**model_hyperparams);
        elif model_type=='RandomForestRegressor':model_production=RandomForestRegressor(**model_hyperparams);

    model_production.fit(X_all_scaled, opened_target);
    #print("Финальная модель обучена на 100% открытых данных.");

    # 8. Генерация ID модели и сохранение Production-модели
    model_id:str=''.join(random.choices(population=string.ascii_uppercase+string.digits,k=16));
    filename:str=f"model_{model_id}.pkl";

    # Сохраняем именно production-модель и production-scaler!
    with open(file=filename,mode='wb') as f:pickle.dump({'model': model_production, 'scaler': scaler_production}, f)
    #print(f"Финальная модель сохранена в файл: {filename}")

    # 9. Логирование
    log_record_txt:str=f"""
--- Model ID: {model_id} ---
Model type: {model_type}
Hyperparameters: {model_hyperparams}
split_random_state: {split_random_state}
Score type: {score_type}
Validation scores: {[f'{s:.6f}' for s in valid_scores]}
Mean validation score: {score_valid_mean:.6f}
Final test score (holdout): {score_test:.6f}
---------------------------------------
"""
    #header_str:str=f'model_id,model_type,score_type,mean_val_score,test_score';
    log_record_csv:str=f'{model_id},{model_type},{score_type},{score_valid_mean},{score_test}\n';
    print(log_record_txt);
    with open(file='log.txt',mode='at',encoding='UTF-8')as log_file:log_file.write(log_record_txt);
    with open(file='log.csv',mode='at',encoding='UTF-8')as log_file:log_file.write(log_record_csv);
    return model_id;

def float_list_to_comma_separated_str(_list,digits:int=2):
    _list = list(np.round(np.array(_list),digits));
    return ','.join([str(x) for x in _list]);

def create_predictions_json(model_ids:list[str],digits:int=2)->None:
    targets_dict:dict[str:float]={};
    for id in closed_ids:
        targets_dict[id]=0.0;
    print(f'targets_dict: {targets_dict}');
    print(f'len(targets_dict): {len(targets_dict)}');
    model_ids_str:str=' '.join(model_ids);
    #=======================Далее код, изначально перемещённый из основного цикла программы (затем доработанный):
    n_models:int=len(model_ids);
    print(f'Предсказания выполняются усреднением результатов для n_models={n_models} моделей со значениями id: {model_ids}');
    buf_s:str='';
    n_samples_closed:int=closed_ids.shape[0];
    targets_list:list[float]=[0.0 for i in range(n_samples_closed)];#Изначально предсказания для всех образцов инициализируем нулями
    #так как 0+a=a для любого действительного числа a, все предсказания - это числа с плавающей точкой
    #Суммирование предсказаний моделей:
    for model_id in model_ids:#Перебираем модели
        pkl_file_name:str=f'model_{model_id}.pkl';
        with open(file=pkl_file_name,mode='rb')as f:model_dict:dict=pickle.load(file=f);
        print(f'model_id: {model_id}, model_dict: {model_dict}');
        model=model_dict['model'];
        scaler=model_dict['scaler'];
        print(f'model_id: {model_id}');
        print(f'model.__dict__: {model.__dict__}');
        print(f'scaler.__dict__: {scaler.__dict__}');
        num_processed:int=0;
        
        for sample_num in range(n_samples_closed):#Перебираем образцы закрытых данных
            id:str=closed_ids[sample_num];
            features:np.ndarray=closed_data[sample_num].reshape(1, -1);#
            features_scaled=scaler.transform(features);
            target_predicted:float=model.predict(features_scaled)[0];
            num_processed=num_processed+1;
            #targets_list.append(target_predicted);
            targets_list[sample_num]=targets_list[sample_num]+target_predicted;
            print(f'sample_num: {sample_num:5d}, id: {id}, target_predicted: {target_predicted}');
    #Усреднение предсказаний моделей:
    for sample_num in range(n_samples_closed):
        targets_list[sample_num]=targets_list[sample_num]/n_models;#Деление суммы предсказаний моделей на количество моделей
        buf_s=buf_s+closed_ids[sample_num]+'\t'+str(targets_list[sample_num])+'\n';#Добавление информации в строку для вывода в tsv файл


    
    tsv_filename:str='result_'+model_ids_str+'.tsv';
    with open(file=tsv_filename,mode='wt',encoding='UTF-8')as tsv_file:tsv_file.write(buf_s);
    
    #targets_ndarray:np.ndarray=np.ndarray(shape=(n_samples_closed,),dtype=np.float32);
    #for i in range(n_samples_closed):targets_ndarray[i]=targets_list[i];
    #targets_ndarray=targets_ndarray.round(decimals=2);
    predictions_str:str=float_list_to_comma_separated_str(_list=targets_list,digits=digits);
    #predictions_str:str=','.join(str(prediction)for prediction in targets_list);
    json_dict:dict={'predictions':predictions_str};
    json_filename:str='result_'+model_ids_str+'.json';
    with open(file=json_filename,mode='wt',encoding='UTF-8')as json_file:json.dump(obj=json_dict,fp=json_file);
    print(f'Модели с id {model_ids_str} применены, их усреднённые результаты в файлах {tsv_filename} и {json_filename}');


    pass;

#Действия, выполняемые перед каждым запуском:
opened_data,opened_target,opened_ids,closed_data,closed_target,closed_ids=load_data();
create_log_files();

#Основной цикл программы:
command_num:int=0;
while command_num>-1:
    print(f'=====================================');
    print(f'1 => выполнить кросс-валидацию n раз v5');
    print(f'2 => создать json файл с предсказанием модели или средним предсказанием нескольких моделей из списка их id');
    print(f'3 => ');

    print(f'-1 => выйти из программы');
    print(f'=====================================');

    input_str:str=input('Введите номер команды: ');
    print(f'Введено: {input_str}');
    command_num=int(input_str);
    if command_num==1:#1 => выполнить кросс-валидацию n раз v5
        num_of_experiments:int=int(input('Введите количество экспериментов: '));
        for i in range(num_of_experiments):
            try:
                print(f'Эксперимент {i+1}/{num_of_experiments}... ',end='');
                model_id:str=run_one_model_experiment_v5(problem_type='regression',task_output='mono_output',score_type='mean_squared_error',
                model_type=None,model_hyperparams=None,num_folds=10,score_valid_min_threshold=None,score_valid_max_threshold=0.10);
            except Exception as ex:
                print(f'Возникло исключение, type(ex): {type(ex)}, ex: {ex}');
    elif command_num==2:#2 => создать json файл с предсказанием модели или средним предсказанием нескольких моделей из списка их id
        model_ids_str:str=input('Введите id модели или нескольких моделей через запятую (например, [08JZRAWXBE5N43MX] или [08JZRAWXBE5N43MX,2352C29OXLDYGPAL,J0KZOWU71FHE3TCR,EENT8VMHI4CK4D24]): ');
        dirits_round:int=int(input('Введите количество цифр округления (например, 2): '));
        model_ids_list:list[str]=model_ids_str.split(sep=',');
        create_predictions_json(model_ids=model_ids_list,digits=dirits_round);
        




        pass;
print(f'Работа программы завершена');



