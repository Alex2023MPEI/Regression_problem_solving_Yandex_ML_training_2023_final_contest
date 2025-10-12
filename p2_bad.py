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

# Добавляем импорты для отбора признаков
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor as RF_for_selection

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

    opened_ids:np.ndarray=np.ndarray(shape=(opened_data.shape[0],),dtype='U4');
    for i in range(opened_data.shape[0]):
        opened_ids[i]=f'{i:04}';
    print(f'opened_ids[0]: {opened_ids[0]}, opened_ids[799]: {opened_ids[799]}');
    print(f'opened_ids.shape: {opened_ids.shape}');

    closed_ids:np.ndarray=np.ndarray(shape=(closed_data.shape[0],),dtype='U4');
    for i in range(closed_data.shape[0]):
        closed_ids[i]=f'{i:04}';
    print(f'closed_ids[0]: {closed_ids[0]}, closed_ids[199]: {closed_ids[199]}');
    print(f'closed_ids.shape: {closed_ids.shape}');

    return opened_data,opened_target,opened_ids,closed_data,closed_target,closed_ids;

def create_log_files()->None:
    """Функция создаёт log файлы (если они не существуют)"""
    if pathlib.Path('log.txt').exists()==False:
        with open(file='log.txt',mode='wt',encoding='UTF-8')as f_log:
            pass;
    if pathlib.Path('log.csv').exists()==False:
        with open(file='log.csv',mode='wt',encoding='UTF-8')as f_log:
            header_str:str=f'model_id,model_type,feature_selector_type,score_type,score_valid_mean,score_test';
            print(header_str,file=f_log);
            pass;
    pass;

def run_one_model_experiment_v6(problem_type:str,task_output:str,score_type:str,fbeta_score_beta:float=1.0,d2_pinball_score_alpha:float=0.5,d2_tweedie_score_power:float=0.0,mean_pinball_loss_alpha:float=0.5,mean_tweedie_deviance_power:float=0.0,model_type:str=None,model_hyperparams:dict=None,feature_selector_type:str=None,feature_selector_params:dict=None,num_folds:int=10,score_valid_min_threshold:float=None,score_valid_max_threshold:float=None)->str:
    """
    Запуск одного эксперимента со случайным выбором модели, отбора признаков и их гиперпараметров
    """
    print(f'Функция run_one_model_experiment_v6 вызвана с параметрами: {locals()}');
    error_str:str='MODEL_WITH_ERROR';
    
    split_random_state:int=int(time.time()*(10**9))%(2**32);
    hyperparam_random_state:int=int(random.uniform(a=0.0,b=1e20))%(2**32);

    opened_data_len:int=opened_data.shape[0];

    if problem_type=='classification':
        X_train_cv,X_test_final,y_train_cv,y_test_final,ids_train_cv,ids_test_final=train_test_split(
        opened_data,opened_target,opened_ids,test_size=0.2,random_state=split_random_state,stratify=opened_target);
    elif problem_type=='regression':
        X_train_cv,X_test_final,y_train_cv,y_test_final,ids_train_cv,ids_test_final=train_test_split(
        opened_data,opened_target,opened_ids,test_size=0.2,random_state=split_random_state);

    print(f'Доля X_train_cv от opened_data: {X_train_cv.shape[0]/opened_data_len}');
    print(f'Доля X_test_final от opened_data: {X_test_final.shape[0]/opened_data_len}');

    # 3. Установка типа и параметров модели
    if problem_type=='classification':
        if model_type is None:
            model_types:list[str]=['AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier','HistGradientBoostingClassifier','RandomForestClassifier','XGBClassifier','LGBMClassifier','LogisticRegression','PassiveAggressiveClassifier','Perceptron','RidgeClassifier','SGDClassifier'];
            model_type:str=random.choice(seq=model_types);
        if model_hyperparams is None:
            # ... (существующий код для классификации)
            pass
    elif problem_type=='regression':
        if model_type is None:
            if task_output=='mono_output':
                model_types:list[str]=['LinearRegression','Ridge','SGDRegressor','ElasticNet','Lars','Lasso','LassoLars','LassoLarsIC','ARDRegression','BayesianRidge','HuberRegressor','QuantileRegressor','RANSACRegressor','TheilSenRegressor','GammaRegressor','PoissonRegressor','TweedieRegressor','PassiveAggressiveRegressor','AdaBoostRegressor','BaggingRegressor','ExtraTreesRegressor','GradientBoostingRegressor','HistGradientBoostingRegressor','RandomForestRegressor'];
            elif task_output=='multi_output':
                model_types:list[str]=['LinearRegression','Ridge','SGDRegressor','ElasticNet','Lars','Lasso','LassoLars','LassoLarsIC','ARDRegression','BayesianRidge','MultiTaskElasticNet','MultiTaskLasso','HuberRegressor','QuantileRegressor','RANSACRegressor','TheilSenRegressor','GammaRegressor','PoissonRegressor','TweedieRegressor','PassiveAggressiveRegressor','AdaBoostRegressor','BaggingRegressor','ExtraTreesRegressor','GradientBoostingRegressor','HistGradientBoostingRegressor','RandomForestRegressor'];
            else:
                print(f'Необходимо задать тип выхода (параметр task_output:str, значения: mono_output или multi_output)');
                return error_str;
            model_type:str=random.choice(seq=model_types);
        if model_hyperparams is None:
            # ... (существующий код для регрессии)
            pass

    # 3.1 Установка типа и параметров отбора признаков
    if feature_selector_type is None:
        feature_selector_types = ['SelectKBest', 'SelectFromModel', 'RFE', 'PCA', 'None']
        feature_selector_type = random.choice(feature_selector_types)
    
    if feature_selector_params is None:
        n_features = opened_data.shape[1]
        if feature_selector_type == 'SelectKBest':
            k = random.randint(max(1, n_features//4), min(n_features, n_features//2))
            score_func = random.choice([f_regression, mutual_info_regression])
            feature_selector_params = {'k': k, 'score_func': score_func}
        elif feature_selector_type == 'SelectFromModel':
            estimator = RandomForestRegressor(n_estimators=50, random_state=hyperparam_random_state)
            threshold = random.choice(['mean', 'median', random.uniform(0.01, 0.5)])
            feature_selector_params = {'estimator': estimator, 'threshold': threshold}
        elif feature_selector_type == 'RFE':
            estimator = RandomForestRegressor(n_estimators=50, random_state=hyperparam_random_state)
            n_features_to_select = random.randint(max(1, n_features//4), min(n_features, n_features//2))
            step = random.randint(1, 3)
            feature_selector_params = {'estimator': estimator, 'n_features_to_select': n_features_to_select, 'step': step}
        elif feature_selector_type == 'PCA':
            n_components = random.randint(max(1, n_features//4), min(n_features-1, n_features//2))
            feature_selector_params = {'n_components': n_components}
        else:  # 'None'
            feature_selector_params = {}

    # 4. Подготовка кросс-валидации
    if problem_type=='classification':
        K_Fold:StratifiedKFold=StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=split_random_state)
    elif problem_type=='regression':
        K_Fold:KFold=KFold(n_splits=num_folds,shuffle=True,random_state=split_random_state);
    
    valid_scores=[];

    print(f"Начинаем кросс-валидацию модели {model_type} с отбором признаков {feature_selector_type}...")
    for fold_num,(train_index,valid_index) in enumerate(K_Fold.split(X=X_train_cv,y=y_train_cv)):
        print(f"  Обрабатываем фолд {fold_num+1}/{num_folds}...",end=' ');

        # 4.1 Разделение на train/valid для фолда
        X_train_fold, X_valid_fold = X_train_cv[train_index], X_train_cv[valid_index]
        y_train_fold, y_valid_fold = y_train_cv[train_index], y_train_cv[valid_index]

        # 4.2 Масштабирование
        scaler_cross_valid:StandardScaler=StandardScaler();
        X_train_fold_scaled = scaler_cross_valid.fit_transform(X_train_fold)
        X_valid_fold_scaled = scaler_cross_valid.transform(X_valid_fold)

        # 4.3 Отбор признаков
        if feature_selector_type == 'SelectKBest':
            feature_selector = SelectKBest(
                score_func=feature_selector_params['score_func'], 
                k=feature_selector_params['k']
            )
        elif feature_selector_type == 'SelectFromModel':
            feature_selector = SelectFromModel(
                estimator=feature_selector_params['estimator'],
                threshold=feature_selector_params['threshold']
            )
        elif feature_selector_type == 'RFE':
            feature_selector = RFE(
                estimator=feature_selector_params['estimator'],
                n_features_to_select=feature_selector_params['n_features_to_select'],
                step=feature_selector_params['step']
            )
        elif feature_selector_type == 'PCA':
            feature_selector = PCA(n_components=feature_selector_params['n_components'])
        else:  # 'None'
            feature_selector = None

        if feature_selector is not None:
            if hasattr(feature_selector, 'fit_transform'):
                X_train_fold_selected = feature_selector.fit_transform(X_train_fold_scaled, y_train_fold)
                X_valid_fold_selected = feature_selector.transform(X_valid_fold_scaled)
            else:
                # Для PCA и подобных
                X_train_fold_selected = feature_selector.fit_transform(X_train_fold_scaled)
                X_valid_fold_selected = feature_selector.transform(X_valid_fold_scaled)
        else:
            X_train_fold_selected = X_train_fold_scaled
            X_valid_fold_selected = X_valid_fold_scaled

        print(f'Признаков после отбора: {X_train_fold_selected.shape[1]}')

        # 4.4 & 4.5 Обучение и оценка
        if problem_type=='classification':
            # ... (существующий код для классификации)
            pass
        elif problem_type=='regression':
            if model_type=='LinearRegression':model_cross_valid=LinearRegression(**model_hyperparams);
            elif model_type=='Ridge':model_cross_valid=Ridge(**model_hyperparams);
            # ... (остальные модели регрессии)
            
        model_cross_valid.fit(X_train_fold_selected, y_train_fold);
        y_valid_pred = model_cross_valid.predict(X_valid_fold_selected);
        
        # Расчет метрики
        if problem_type=='regression':
            if score_type=='mean_squared_error':score_valid=mean_squared_error(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='r2_score':score_valid=r2_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            # ... (другие метрики)
        
        print(f"Метрика на фолде {fold_num+1}: {score_valid:.6f}");
        if score_valid_min_threshold!=None and score_valid<score_valid_min_threshold:
            print(f'Метрика ниже порога ({score_valid_min_threshold:.6f}), прерываем валидацию');
            return error_str;
        elif score_valid_max_threshold!=None and score_valid>score_valid_max_threshold:
            print(f'Метрика выше порога ({score_valid_max_threshold:.6f}), прерываем валидацию');
            return error_str;
        valid_scores.append(score_valid);

    # 5. Расчет средней точности по CV
    score_valid_mean=np.mean(valid_scores);
    print(f"\nСреднее значение метрики {score_type} по кросс-валидации: {score_valid_mean:.6f}");

    # 6. Оценка качества на отложенной выборке
    scaler_for_final_test:StandardScaler=StandardScaler();
    X_train_cv_scaled = scaler_for_final_test.fit_transform(X_train_cv)
    X_test_final_scaled = scaler_for_final_test.transform(X_test_final)

    # Отбор признаков для финального теста
    if feature_selector_type != 'None':
        if hasattr(feature_selector, 'fit'):
            if hasattr(feature_selector, 'fit_transform'):
                X_train_cv_selected = feature_selector.fit_transform(X_train_cv_scaled, y_train_cv)
                X_test_final_selected = feature_selector.transform(X_test_final_scaled)
            else:
                X_train_cv_selected = feature_selector.fit_transform(X_train_cv_scaled)
                X_test_final_selected = feature_selector.transform(X_test_final_scaled)
        else:
            # Пересоздаем selector для финального обучения
            if feature_selector_type == 'SelectKBest':
                feature_selector_final = SelectKBest(
                    score_func=feature_selector_params['score_func'], 
                    k=feature_selector_params['k']
                )
            elif feature_selector_type == 'SelectFromModel':
                feature_selector_final = SelectFromModel(
                    estimator=feature_selector_params['estimator'],
                    threshold=feature_selector_params['threshold']
                )
            elif feature_selector_type == 'RFE':
                feature_selector_final = RFE(
                    estimator=feature_selector_params['estimator'],
                    n_features_to_select=feature_selector_params['n_features_to_select'],
                    step=feature_selector_params['step']
                )
            elif feature_selector_type == 'PCA':
                feature_selector_final = PCA(n_components=feature_selector_params['n_components'])
            
            if hasattr(feature_selector_final, 'fit_transform'):
                X_train_cv_selected = feature_selector_final.fit_transform(X_train_cv_scaled, y_train_cv)
                X_test_final_selected = feature_selector_final.transform(X_test_final_scaled)
            else:
                X_train_cv_selected = feature_selector_final.fit_transform(X_train_cv_scaled)
                X_test_final_selected = feature_selector_final.transform(X_test_final_scaled)
    else:
        X_train_cv_selected = X_train_cv_scaled
        X_test_final_selected = X_test_final_scaled

    # Обучение и предсказание на финальном тесте
    if problem_type=='regression':
        if model_type=='LinearRegression':model_for_final_test=LinearRegression(**model_hyperparams);
        elif model_type=='Ridge':model_for_final_test=Ridge(**model_hyperparams);
        # ... (остальные модели)
        
    model_for_final_test.fit(X_train_cv_selected, y_train_cv);
    y_test_pred = model_for_final_test.predict(X_test_final_selected);
    
    if problem_type=='regression':
        if score_type=='mean_squared_error':score_test=mean_squared_error(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='r2_score':score_test=r2_score(y_true=y_test_final,y_pred=y_test_pred);
        # ... (другие метрики)
    
    print(f"Метрика на тесте: {score_test:.6f}");
    
    # 7. ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ НА 100% ОТКРЫТЫХ ДАННЫХ
    print("Обучение финальной production-модели на ВСЕХ данных...")
    scaler_production:StandardScaler=StandardScaler();
    X_all_scaled = scaler_production.fit_transform(opened_data)
    
    # Отбор признаков для production модели
    if feature_selector_type != 'None':
        if feature_selector_type == 'SelectKBest':
            feature_selector_production = SelectKBest(
                score_func=feature_selector_params['score_func'], 
                k=feature_selector_params['k']
            )
        elif feature_selector_type == 'SelectFromModel':
            feature_selector_production = SelectFromModel(
                estimator=feature_selector_params['estimator'],
                threshold=feature_selector_params['threshold']
            )
        elif feature_selector_type == 'RFE':
            feature_selector_production = RFE(
                estimator=feature_selector_params['estimator'],
                n_features_to_select=feature_selector_params['n_features_to_select'],
                step=feature_selector_params['step']
            )
        elif feature_selector_type == 'PCA':
            feature_selector_production = PCA(n_components=feature_selector_params['n_components'])
        
        if hasattr(feature_selector_production, 'fit_transform'):
            X_all_selected = feature_selector_production.fit_transform(X_all_scaled, opened_target)
        else:
            X_all_selected = feature_selector_production.fit_transform(X_all_scaled)
    else:
        feature_selector_production = None
        X_all_selected = X_all_scaled

    # Обучение production модели
    if problem_type=='regression':
        if model_type=='LinearRegression':model_production=LinearRegression(**model_hyperparams);
        elif model_type=='Ridge':model_production=Ridge(**model_hyperparams);
        # ... (остальные модели)
        
    model_production.fit(X_all_selected, opened_target);

    # 8. Генерация ID модели и сохранение
    model_id:str=''.join(random.choices(population=string.ascii_uppercase+string.digits,k=16));
    filename:str=f"model_{model_id}.pkl";

    # Сохраняем production-модель, scaler и feature_selector
    with open(file=filename,mode='wb') as f:
        pickle.dump({
            'model': model_production, 
            'scaler': scaler_production, 
            'feature_selector': feature_selector_production,
            'feature_selector_type': feature_selector_type,
            'feature_selector_params': feature_selector_params
        }, f)

    # 9. Логирование
    log_record_txt:str=f"""
--- Model ID: {model_id} ---
Model type: {model_type}
Feature selector type: {feature_selector_type}
Feature selector params: {feature_selector_params}
Hyperparameters: {model_hyperparams}
split_random_state: {split_random_state}
Score type: {score_type}
Validation scores: {[f'{s:.6f}' for s in valid_scores]}
Mean validation score: {score_valid_mean:.6f}
Final test score (holdout): {score_test:.6f}
---------------------------------------
"""
    log_record_csv:str=f'{model_id},{model_type},{feature_selector_type},{score_type},{score_valid_mean},{score_test}\n';
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
    
    n_models:int=len(model_ids);
    print(f'Предсказания выполняются усреднением результатов для n_models={n_models} моделей со значениями id: {model_ids}');
    buf_s:str='';
    n_samples_closed:int=closed_ids.shape[0];
    targets_list:list[float]=[0.0 for i in range(n_samples_closed)];
    
    for model_id in model_ids:
        pkl_file_name:str=f'model_{model_id}.pkl';
        with open(file=pkl_file_name,mode='rb')as f:
            model_dict:dict=pickle.load(file=f);
        print(f'model_id: {model_id}');
        
        model=model_dict['model'];
        scaler=model_dict['scaler'];
        feature_selector=model_dict['feature_selector'];
        
        num_processed:int=0;
        
        for sample_num in range(n_samples_closed):
            id:str=closed_ids[sample_num];
            features:np.ndarray=closed_data[sample_num].reshape(1, -1);
            features_scaled=scaler.transform(features);
            
            # Применяем отбор признаков если он есть
            if feature_selector is not None:
                features_selected = feature_selector.transform(features_scaled)
            else:
                features_selected = features_scaled
                
            target_predicted:float=model.predict(features_selected)[0];
            num_processed=num_processed+1;
            targets_list[sample_num]=targets_list[sample_num]+target_predicted;
            print(f'sample_num: {sample_num:5d}, id: {id}, target_predicted: {target_predicted}');
    
    #Усреднение предсказаний моделей:
    for sample_num in range(n_samples_closed):
        targets_list[sample_num]=targets_list[sample_num]/n_models;
        buf_s=buf_s+closed_ids[sample_num]+'\t'+str(targets_list[sample_num])+'\n';

    tsv_filename:str='result_'+model_ids_str+'.tsv';
    with open(file=tsv_filename,mode='wt',encoding='UTF-8')as tsv_file:tsv_file.write(buf_s);
    
    predictions_str:str=float_list_to_comma_separated_str(_list=targets_list,digits=digits);
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
    print(f'1 => выполнить кросс-валидацию n раз v6 (с отбором признаков)');
    print(f'2 => создать json файл с предсказанием модели или средним предсказанием нескольких моделей из списка их id');
    print(f'3 => ');

    print(f'-1 => выйти из программы');
    print(f'=====================================');

    input_str:str=input('Введите номер команды: ');
    print(f'Введено: {input_str}');
    command_num=int(input_str);
    if command_num==1:
        num_of_experiments:int=int(input('Введите количество экспериментов: '));
        for i in range(num_of_experiments):
            try:
                print(f'Эксперимент {i+1}/{num_of_experiments}... ',end='');
                model_id:str=run_one_model_experiment_v6(
                    problem_type='regression',
                    task_output='mono_output',
                    score_type='mean_squared_error',
                    model_type=None,
                    model_hyperparams=None,
                    feature_selector_type=None,
                    feature_selector_params=None,
                    num_folds=10,
                    score_valid_min_threshold=None,
                    score_valid_max_threshold=0.10
                );
            except Exception as ex:
                print(f'Возникло исключение, type(ex): {type(ex)}, ex: {ex}');
    elif command_num==2:
        model_ids_str:str=input('Введите id модели или нескольких моделей через запятую: ');
        dirits_round:int=int(input('Введите количество цифр округления (например, 2): '));
        model_ids_list:list[str]=model_ids_str.split(sep=',');
        create_predictions_json(model_ids=model_ids_list,digits=dirits_round);

print(f'Работа программы завершена');