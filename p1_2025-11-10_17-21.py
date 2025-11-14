#Ссылка на контест для этой задачи (assignment_final): https://contest.yandex.ru/contest/56809/problems/
import numpy as np;
import datetime,json,os,pathlib,pickle,random,string,time,typing;
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler,StandardScaler;
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold;
from sklearn.impute import KNNImputer,SimpleImputer,MissingIndicator;
#Импорты для классификации:
from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier,Perceptron,RidgeClassifier,SGDClassifier;
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier;
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier;
from xgboost import XGBClassifier;from lightgbm import LGBMClassifier;
from sklearn.metrics import accuracy_score,auc,average_precision_score,balanced_accuracy_score,brier_score_loss;
from sklearn.metrics import cohen_kappa_score,dcg_score,f1_score,fbeta_score,hamming_loss,hinge_loss,jaccard_score;
from sklearn.metrics import log_loss,matthews_corrcoef,ndcg_score,precision_score,recall_score,roc_auc_score,zero_one_loss;
#Импорты для регрессии:
from sklearn.linear_model import LinearRegression,Ridge,SGDRegressor,ElasticNet,Lasso,LassoLarsIC,ARDRegression,OrthogonalMatchingPursuit;
from sklearn.linear_model import BayesianRidge,MultiTaskElasticNet,MultiTaskLasso,HuberRegressor,QuantileRegressor,RANSACRegressor,Lars;
from sklearn.linear_model import TheilSenRegressor,GammaRegressor,PoissonRegressor,TweedieRegressor,PassiveAggressiveRegressor,LassoLars;
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor;
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor;
from sklearn.metrics import d2_absolute_error_score,d2_pinball_score,d2_tweedie_score,explained_variance_score,max_error;
from sklearn.metrics import mean_absolute_percentage_error,mean_gamma_deviance,mean_pinball_loss,mean_poisson_deviance;
from sklearn.metrics import mean_squared_error,mean_squared_log_error,mean_tweedie_deviance,median_absolute_error;
from sklearn.metrics import r2_score,root_mean_squared_error,root_mean_squared_log_error;
#Импорты для feature_selector (классы отбора признаков):
from sklearn.feature_selection import GenericUnivariateSelect,RFE,RFECV,SelectFdr,SelectFpr,SelectFromModel,SelectFwe,SelectKBest;
from sklearn.feature_selection import SelectPercentile,SequentialFeatureSelector,VarianceThreshold;
#Импорты для feature_selector (функции для score_func для SelectFdr,SelectFpr,SelectFwe,SelectKBest,SelectPercentile):
from sklearn.feature_selection import chi2,f_classif,f_regression,mutual_info_classif,mutual_info_regression,r_regression;
from sklearn.svm import LinearSVC,LinearSVR,NuSVC,NuSVR,SVC,SVR;

AnyImputer:typing.TypeAlias=KNNImputer|SimpleImputer;#IterativeImputer is experimental and the API might change without any deprecation cycle.
AnyScaler:typing.TypeAlias=MaxAbsScaler|MinMaxScaler|RobustScaler|StandardScaler;
AnyFSEstimator:typing.TypeAlias=LogisticRegression|PassiveAggressiveClassifier|Perceptron|RidgeClassifier|SGDClassifier|LinearRegression|Ridge|SGDRegressor|ElasticNet|Lars|Lasso|LassoLars|LassoLarsIC|OrthogonalMatchingPursuit|ARDRegression|BayesianRidge|HuberRegressor|QuantileRegressor|RANSACRegressor|TheilSenRegressor|GammaRegressor|PoissonRegressor|TweedieRegressor|PassiveAggressiveRegressor|LinearSVC|NuSVC|SVC|LinearSVR|NuSVR|SVR;
AnyFeatureSelector:typing.TypeAlias=GenericUnivariateSelect|RFE|RFECV|SelectFdr|SelectFpr|SelectFromModel|SelectFwe|SelectKBest|SelectPercentile|SequentialFeatureSelector;
AnyModel:typing.TypeAlias=LogisticRegression|PassiveAggressiveClassifier|Perceptron|RidgeClassifier|SGDClassifier|AdaBoostClassifier|BaggingClassifier|ExtraTreesClassifier|GradientBoostingClassifier|HistGradientBoostingClassifier|RandomForestClassifier|XGBClassifier|LGBMClassifier|LinearRegression|Ridge|SGDRegressor|ElasticNet|Lasso|LassoLarsIC|ARDRegression|OrthogonalMatchingPursuit|BayesianRidge|MultiTaskElasticNet|MultiTaskLasso|HuberRegressor|QuantileRegressor|RANSACRegressor|Lars|TheilSenRegressor|GammaRegressor|PoissonRegressor|TweedieRegressor|PassiveAggressiveRegressor|LassoLars|AdaBoostRegressor|BaggingRegressor|ExtraTreesRegressor|GradientBoostingRegressor|HistGradientBoostingRegressor|RandomForestRegressor;

def true_with_prob(p:float=0.5)->bool:
    """Функция возвращает True с вероятностью p и False с вероятностью q=1-p"""
    if isinstance(p,float)==False:#Если вдруг по ошибке передано не то значение
        p:float=0.5;
    if p<0.0:p=0.0;
    if p>1.0:p=1.0;
    q:float=1.0-p;
    rand_num:float=random.random();#Return the next random floating-point number in the range 0.0<=X<1.0
    if rand_num<p:return True;
    else:return False;

def str_to_float(s:str='1.0',num_min:float=0.0,num_max:float=1.0,num_default:float=1.0)->float:
    """Функция преобразует строку s в число типа float с корретной обработкой возможных исключений"""
    try:
        num:float=float(s);
    except:
        num:float=num_default;
        print(f'Строку [{s}] невозможно преобразовать к типу float, вместо этого возвращено значение {num}');
    if num<num_min:
        print(f'Число num={num} меньше чем num_min={num_min}, поэтому возвращено число {num_min}');
        num=num_min;
    if num>num_max:
        print(f'Число num={num} больше чем num_max={num_max}, поэтому возвращено число {num_max}');
        num=num_max;
    return num;

#ОЧЕНЬ ПОЛЕЗНАЯ ФУНКЦИЯ (список словарей в Python в некотором смысле аналогичен массиву записей в Delphi)
def read_csv(filename:str,delimiter_values:str=',')->list[dict]:#должна возвращать список словарей
    with open(file=filename,mode='rt',encoding='utf-8')as f:lst_lines=[s.rstrip()for s in f.readlines()];
    lst_of_dicts:list[dict]=[];#Изначально список словарей пустой
    lst_keys:list[str]=lst_lines[0].split(delimiter_values);#Первая строка csv файла - это ключи
    lst_values:list[list[str]]=[lst_lines[line_num].split(delimiter_values)for line_num in range(1,len(lst_lines))];
    for dict_num in range(len(lst_lines)-1):lst_of_dicts.append(dict(tuple(zip(lst_keys,lst_values[dict_num]))));
    return(lst_of_dicts);
#И обратная ей функция (записывает список словарей в csv файл)
def write_csv(filename:str,delimiter_values:str,lst_of_dicts:list[dict])->None:#должна принимать список словарей и записывать в *.csv
    s:str=delimiter_values.join(list(lst_of_dicts[0].keys()))+'\n';#Сначала записываем в строку ключи (названия столбцов)
    for d in lst_of_dicts:s=s+delimiter_values.join(d.values())+'\n';#Затем в цикле добавляем в строку значения
    with open(file=filename,mode='wt',encoding='utf-8')as f:f.write(s);
    return None;

def analize_log_pipelines_csv(log_pipelines_csv_file_name:str='log_pipelines.csv',score_valid_mean_threshold_min:float=None,score_valid_mean_threshold_max:float=None,score_test_threshold_min:float=None,score_test_threshold_max:float=None,n_features_selected_randomly_threshold_min:float=None,n_features_selected_randomly_threshold_max:float=None)->None:
    """Анализ csv файла для определения id лучших пайплайнов (для которых выполняются заданные ограничения)"""
    log_pipelines_dicts_list:list[dict]=read_csv(filename=log_pipelines_csv_file_name,delimiter_values=',');
    #Преобразование значений некоторых полей из типа str в типы int или float
    if'n_features_all'in log_pipelines_dicts_list[0].keys():
        for pipeline_dict in log_pipelines_dicts_list:pipeline_dict['n_features_all']=int(pipeline_dict['n_features_all']);
    if'n_features_selected_randomly'in log_pipelines_dicts_list[0].keys():
        for pipeline_dict in log_pipelines_dicts_list:pipeline_dict['n_features_selected_randomly']=int(pipeline_dict['n_features_selected_randomly']);
    if'score_valid_mean'in log_pipelines_dicts_list[0].keys():
        for pipeline_dict in log_pipelines_dicts_list:pipeline_dict['score_valid_mean']=float(pipeline_dict['score_valid_mean']);
    if'score_test'in log_pipelines_dicts_list[0].keys():
        for pipeline_dict in log_pipelines_dicts_list:pipeline_dict['score_test']=float(pipeline_dict['score_test']);
    print(f'log_pipelines_dicts_list: {log_pipelines_dicts_list}');
    print(f'len(log_pipelines_dicts_list): {len(log_pipelines_dicts_list)}');
    numeric_keys:list[str]=['n_features_all','n_features_selected_randomly','score_valid_mean','score_test'];
    numeric_keys_stats:list[dict[str:float]]=[];
    for i in range(len(numeric_keys)):
        numeric_keys_stats.append({});
        name:str=numeric_keys[i];
        numeric_keys_stats[i]['name']=numeric_keys[i];
        numeric_keys_stats[i]['number']=sum([1 for j in range(len(log_pipelines_dicts_list))]);
        numeric_keys_stats[i]['sum']=sum([log_pipelines_dicts_list[j][name]for j in range(len(log_pipelines_dicts_list))]);
        numeric_keys_stats[i]['mean']=numeric_keys_stats[i]['sum']/numeric_keys_stats[i]['number'];
        numeric_keys_stats[i]['min']=min([log_pipelines_dicts_list[j][name]for j in range(len(log_pipelines_dicts_list))]);
        numeric_keys_stats[i]['max']=max([log_pipelines_dicts_list[j][name]for j in range(len(log_pipelines_dicts_list))]);
        numeric_keys_stats[i]['std']=(sum([(log_pipelines_dicts_list[j][name]-numeric_keys_stats[i]['mean'])**2 for j in range(len(log_pipelines_dicts_list))])/numeric_keys_stats[i]['number'])**0.5;
        print(f'i: {i}, numeric_keys_stats[i]: {numeric_keys_stats[i]}');
        pass;
    for name in ['n_features_selected_randomly','score_valid_mean','score_test']:
        min_sub_list:list[dict]=sorted(log_pipelines_dicts_list,key=lambda d:d[name],reverse=False)[:10];
        print(f'Пайплайны с наименьшими значениями {name}:');
        for i in range(len(min_sub_list)):print(f'{i}) {min_sub_list[i]}');
        max_sub_list:list[dict]=sorted(log_pipelines_dicts_list,key=lambda d:d[name],reverse=True)[:10];
        print(f'Пайплайны с наибольшими значениями {name}:');
        for i in range(len(max_sub_list)):print(f'{i}) {max_sub_list[i]}');
    s_lst:list[dict]=[d for d in log_pipelines_dicts_list];#s_lst - список только тех пайплайнов, у которых числовые значения удовлетворяют ограничениям
    if score_valid_mean_threshold_min is not None:s_lst:list[dict]=[d for d in s_lst if d['score_valid_mean']>=score_valid_mean_threshold_min];
    if score_valid_mean_threshold_max is not None:s_lst:list[dict]=[d for d in s_lst if d['score_valid_mean']<=score_valid_mean_threshold_max];
    if score_test_threshold_min is not None:s_lst:list[dict]=[d for d in s_lst if d['score_test']>=score_test_threshold_min];
    if score_test_threshold_max is not None:s_lst:list[dict]=[d for d in s_lst if d['score_test']<=score_test_threshold_max];
    if n_features_selected_randomly_threshold_min is not None:s_lst:list[dict]=[d for d in s_lst if d['n_features_selected_randomly']>=n_features_selected_randomly_threshold_min];
    if n_features_selected_randomly_threshold_max is not None:s_lst:list[dict]=[d for d in s_lst if d['n_features_selected_randomly']<=n_features_selected_randomly_threshold_max];
    print(f'Список отобранных пайплайнов:');
    for d in s_lst:print(d);
    print(f'В списке отобранных пайплайнов {len(s_lst)} пайплайнов из {len(log_pipelines_dicts_list)} ({(100*len(s_lst)/len(log_pipelines_dicts_list)):.4f}%)');
    print(f'Пайплайны отобраны с ограничениями: {locals()}');
    ids_of_selected_pipelines:list[str]=sorted([d['pipeline_id']for d in s_lst]);
    print(f'id выбранных {len(ids_of_selected_pipelines)} пайплайнов: [{" ".join(ids_of_selected_pipelines)}]');
    pass;

def analize_log_pipelines_txt(log_pipelines_txt_file_name:str='log_pipelines.txt',num_of_most_times_used_features_indexes:int=14)->None:
    """Функция проводит анализ файла log_pipelines.txt и выявляет, какие признаки (по индексам) наиболее часто использовались в лучших
    пайплайнах. Это полезно в том случае, если необходимо построить пайплайн с использованием не более чем некоторого количества признаков
    (например, в задаче [B. Финальное соревнование: задача 2], где в условии сказано: [Вторая модель должна быть линейной, т.е.
    представлять собой линейную комбинацию признаков плюс смещение, модель не должна использовать более 15 параметров (14 весов плюс
    смещение)])\n
    При таком условии:
    1. Определяем лучшие признаки (условно, если признак номер 15 использован в 100 лучших пайплайнах, а признак номер 23 использован в
    60 лучших пайплайнах, то наверное признак номер 15 полезнее, чем признак номер 23). Для каждого признака (его индекса) опредеяем,
    количество раз, сколько этот признак использован в лучших пайплайнах, затем сортируем по убыванию этих количеств и отбираем 14 тех
    признаков, которые использованы в наибольшем количестве моделей (именно 14 признаков, так как установлено ограничение в 15
    параметров, один из которых - это смещение [bias])
    2. Выполняем построение пайплайнов с отбором конкретно этих 14 признаков (использование различных пайплайнов, RandomSearch для подбора
    гиперпараметров, кросс-валидация на 10 фолдов, тестирование на отложенной выборке, обучение прошедших порог пайплайнов на всех
    открытых данных, их сохранение в *.pkl файлы вместе со Scaler и запись результатов в txt и csv логи)
    3. Затем отбор пайплайнов, удоветворяющих пороговым значениям score_valid_mean и score_test (функция analize_log_pipelines_csv) и
    усреднение их предсказаний. Каждая модель линейная и имеет 14 коэффициентов k0,...,k13 + bias следовательно для усреднения
    их предсказаний усредняем их коэффициенты k0,...,k13 и bias. Например, если отобрано 100 лучших пайплайнов, значит
    k0=(k0[0]+k0[1]+..+k0[99])/100, k1=(k1[0]+k1[1]+..+k1[99])/100, ..., k13=(k13[0]+k13[1]+..+k13[99])/100,
    bias=(bias[0]+bias[1]+..+bias[99])/100
    Нужно будет ещё подумать над scaler для каждого из 14 признаков, но тут скорее всего Scaler у всех моделей будет одинаковый
    (так как Scaler не зависит от того, какая после него применена модель + каждая итоговая модель [сохраняемая затем в *.pkl файл]
    обучается на всех 100% открытых данных, поэтому очевидно, что при задании фиксированного списка лучших признаков [randomly_selected_indexes]
    scaler у моделей во всех *.pkl файлах должны быть одинаковые).\n
    После применения функции analize_one_pkl_file к двум pkl файлам моделей подтверждено, что всё содержимое scaler у разных моделей
    полностью совпадает, например: value.__dict__: {'with_mean': True, 'with_std': True, 'copy': True, 'n_features_in_': 392, 'n_samples_seen_': 800, 'mean_': array([-5.62410914e-01,  2.58281096e-01, -5.68630481e-01, -6.69724427e-04,...
    \n
    Равенство Scaler для разных моделей - это правильно, но почему-то и model и scaler выдают: 'n_features_in_': 392
    \nP.S. 'n_features_in_': 392 - это было следствием ошибки в начале код функции run_one_pipeline_experiment_v1 в части: 
    отбор num_features_select_from_all признаков из всех. Из-за этой ошибки если максимальное и минимальное количества используемых
    признаков заданы как нули, то даже при заданном списке индексов в массивы opened_data и closed_data попадали все признаки (392
    признака в этой задаче). Именно из-за такого огромного количества признаков при проведении эксперимента много раз за почти сутки
    появилось только около 40 +-хороших моделей. Теперь эта ошибка ИСПРАВЛЕНА. Теперь сначала проверяется наличие списка индексов
    признаков, затем если его нет, то список индексов признаков заполняется.

    \nЕсли ограничение на количество признаков не установлено, то вероятно эта функция не очень нужна"""
    with open(file=log_pipelines_txt_file_name,mode='rt',encoding='UTF-8')as f:
        txt_log_lines:list[str]=[line.rstrip('\n') for line in f.readlines()];
    print(f'len(txt_log_lines): {len(txt_log_lines)}');
    print(f'txt_log_lines[4]: {txt_log_lines[4]}');
    randomly_selected_indexes_lines:list[str]=[line for line in txt_log_lines if 'randomly_selected_indexes: 'in line];
    randomly_selected_indexes_lines=[line.replace('randomly_selected_indexes: [','').replace(']',',').replace(' ','')for line in randomly_selected_indexes_lines];
    print(f'len(randomly_selected_indexes_lines): {len(randomly_selected_indexes_lines)}');
    print(f'randomly_selected_indexes_lines[:10]: {randomly_selected_indexes_lines[:10]}');
    n_features_selected_randomly_lines:list[str]=[line for line in txt_log_lines if 'n_features_selected_randomly: 'in line];
    n_features_selected_randomly_lines=[line.replace('n_features_selected_randomly: ','') for line in n_features_selected_randomly_lines];
    n_features_selected_randomly_ints:list[int]=[int(line)for line in n_features_selected_randomly_lines];
    print(f'len(n_features_selected_randomly_ints): {len(n_features_selected_randomly_ints)}');
    print(f'n_features_selected_randomly_ints[:50]: {n_features_selected_randomly_ints[:50]}');
    print(f'sum(n_features_selected_randomly_ints): {sum(n_features_selected_randomly_ints)}');
    randomly_selected_indexes_str:str=''.join(randomly_selected_indexes_lines);
    randomly_selected_indexes_list:list[int]=[int(num)for num in randomly_selected_indexes_str.split(sep=',')if len(num)>0];#Чтобы не пытаться
    #преобразовать пустую строку после последней запятой в число типа int
    print(f'len(randomly_selected_indexes_list): {len(randomly_selected_indexes_list)}');
    print(f'randomly_selected_indexes_list[:50]: {randomly_selected_indexes_list[:50]}');
    if sum(n_features_selected_randomly_ints)==len(randomly_selected_indexes_list):
        print(f'sum(n_features_selected_randomly_ints): {sum(n_features_selected_randomly_ints)}, len(randomly_selected_indexes_list): {len(randomly_selected_indexes_list)}, эти числа равны, проверка работает правильно');
    else:#Сумма n_features_selected_randomly_ints должна быть равна длине randomly_selected_indexes_list (и равна суммарному количеству выбранных признаков во всех пайплайнах, попавших в файл log_pipelines.txt)
        print(f'sum(n_features_selected_randomly_ints): {sum(n_features_selected_randomly_ints)}, len(randomly_selected_indexes_list): {len(randomly_selected_indexes_list)}, эти числа НЕ равны, проверка показывает наличие ошибки');
    selected_non_zero_times_indexes:list[int]=sorted(list(set(randomly_selected_indexes_list)));
    print(f'selected_non_zero_times_indexes: {selected_non_zero_times_indexes}, len(selected_non_zero_times_indexes): {len(selected_non_zero_times_indexes)}');
    #Сохранение в словарь {index:num_of_this_index}
    indexes_times_dict:dict[int:int]={};#Ключи - индексы, значения - их количества
    for ind in selected_non_zero_times_indexes:indexes_times_dict[ind]=0;
    for i in randomly_selected_indexes_list:indexes_times_dict[i]=indexes_times_dict[i]+1;
    print(f'indexes_times_dict: {indexes_times_dict}');
    #Сохранение в список словарей [{'index':,'times':},...,{'index':,'times':}]
    indexes_times_list_dicts:list[dict[str:int]]=[];
    for ind in selected_non_zero_times_indexes:indexes_times_list_dicts.append({'index':ind,'times':0});
    #Это рабочий вариант, но лучше так не делать, так как он рассчитывает на то, что словари для всех выбранных индексов
    #расположены по порядку увеличения этих индексов без пропусков
    #for ind in randomly_selected_indexes_list:indexes_times_list_dicts[ind]['times']=indexes_times_list_dicts[ind]['times']+1;
    #Другой вариант (тоже рабочий, но должен быть более общим):
    for ind in randomly_selected_indexes_list:
        index:int=None;
        for index_key in range(len(indexes_times_list_dicts)):#Для эффективности можно заменить это for на while но число признаков
            if indexes_times_list_dicts[index_key]['index']==ind:#вряд ли будет больше ста тысяч, поэтому можно так оставить
                index=ind;
        print(f'len(indexes_times_list_dicts): {len(indexes_times_list_dicts)}, index: {index}');
        indexes_times_list_dicts[index]['times']=indexes_times_list_dicts[index]['times']+1;
    print(f'indexes_times_list_dicts (before sorting): {indexes_times_list_dicts}');
    indexes_times_list_dicts.sort(key=lambda d:d['times'],reverse=True);#Сортировка по убыванию количеств
    print(f'indexes_times_list_dicts (after sorting): {indexes_times_list_dicts}');
    #[{'index':218,'times':114},{'index':307,'times':113},...,{'index':290,'times':71},{'index':269,'times': 69}]
    print(f'{num_of_most_times_used_features_indexes} наиболее часто использованных признаков (в виде словаря номер:количество): {indexes_times_list_dicts[:num_of_most_times_used_features_indexes]}');
    print(f'{num_of_most_times_used_features_indexes} наиболее часто использованных признаков (в виде списка): {[d["index"] for d in indexes_times_list_dicts[:num_of_most_times_used_features_indexes]]}');
    #14 наиболее часто использованных признаков (в виде словаря номер:количество): [{'index': 218, 'times': 114}, {'index': 307, 'times': 113}, {'index': 56, 'times': 111}, {'index': 266, 'times': 111}, {'index': 63, 'times': 110}, {'index': 67, 'times': 110}, {'index': 77, 'times': 109}, {'index': 336, 'times': 108}, {'index': 84, 'times': 107}, {'index': 105, 'times': 107}, {'index': 376, 'times': 106}, {'index': 59, 'times': 105}, {'index': 73, 'times': 105}, {'index': 257, 'times': 105}]
    #14 наиболее часто использованных признаков (в виде списка): [218, 307, 56, 266, 63, 67, 77, 336, 84, 105, 376, 59, 73, 257]

def analize_one_pkl_file(pkl_file_name:str)->None:
    """Функция выводит информацию о содержимом одного pkl файла"""
    with open(file=pkl_file_name,mode='rb')as pkl_file:#binary mode doesn't take an encoding argument
        print(f'======== Информация об одном pkl файле ========:');
        pkl_file_size:int=os.path.getsize(filename=pkl_file_name);
        print(f'pkl_file_name: {pkl_file_name}, pkl_file_size: {pkl_file_size}');        
        pkl_obj=pickle.load(file=pkl_file);
        print(f'pkl_obj: {pkl_obj}');
        print(f'type(pkl_obj): {type(pkl_obj)}');
        if type(pkl_obj)==list:pass;
        elif type(pkl_obj)==dict:
            pkl_obj_keys=pkl_obj.keys();
            print(f'pkl_obj_keys: {pkl_obj_keys}');
            for key in pkl_obj_keys:
                value=pkl_obj[key];
                __dict__str:str=f'Объект {value} не имеет атрибута __dict__';
                if hasattr(value,'__dict__'):__dict__str=f'value.__dict__: {value.__dict__}';
                print(f'key: {key}, type(key): {type(key)}, value: {value}, type(value): {type(value)}, \n value.__dir__(): {value.__dir__()}, \n {__dict__str}\n');

def data_transformation_pairwise_multiplications(feature_matrix:np.ndarray)->np.ndarray:#Функция - трансформация признаков
    """Функция добавляет к матрице признаков все попарные произведения исходных признаков"""
    n_samples:int=feature_matrix.shape[0] ;#количество строк таблицы, остаётся постоянным при добавлении новых признаков
    n_features:int=feature_matrix.shape[1];#количество столбцов таблицы, увеличивается при добавлении новых признаков
    n_pairs:int=n_features*(n_features-1)//2;#количество попарных произведений исходных признаков
    new_feature_matrix:np.ndarray=np.zeros((n_samples,n_features+n_pairs));#Добавление признаков (столбцов), заполненных нулями
    new_feature_matrix[:,:n_features]=feature_matrix;#Копирование значений имеющихся столбцов
    current_col:int=n_features;#Номер текущего столбца для записи нового признака (начинается с n_features)
    for i in range(n_features):
        for j in range(i+1,n_features):
            new_feature_matrix[:,current_col]=feature_matrix[:,i]*feature_matrix[:,j];
            current_col=current_col+1;
    return new_feature_matrix;
def data_transformation_add_functions(feature_matrix:np.ndarray)->np.ndarray:
    """Функция добавляет к матрице признаков результаты применения различных функций к исходным признакам"""
    n_samples:int=feature_matrix.shape[0] ;#Количество строк таблицы, остаётся постоянным при добавлении новых признаков
    n_features:int=feature_matrix.shape[1];#Количество столбцов таблицы, увеличивается при добавлении новых признаков
    n_added_features:int=n_features*13;#Количество добавленных признаков (по 13 функций на каждый признак)
    new_feature_matrix:np.ndarray=np.zeros((n_samples,n_features+n_added_features));#Добавление признаков (столбцов), заполненных нулями
    new_feature_matrix[:,:n_features]=feature_matrix;#Копирование значений имеющихся столбцов
    current_col:int=n_features;#Начинаем добавлять с номера столбца n_features (номера 0..n_features-1 уже заняты)
    for i in range(n_features):
        x:np.ndarray=feature_matrix[:,i];        
        new_feature_matrix[:,current_col+0]=np.exp(x);              #1. exp(x)=e^x (экспонента)
        new_feature_matrix[:,current_col+1]=1.0/(1.0+np.exp(-x));   #2. sigmoid(x)=1/(1+e^(-x)) (сигмоида)
        new_feature_matrix[:,current_col+2]=x**2;                   #3. sqr(x)=x^2 (возведение в квадрат)
        new_feature_matrix[:,current_col+3]=np.abs(x);              #4. abs(x)=|x| (модуль числа)
        new_feature_matrix[:,current_col+4]=np.maximum(0,x);        #5. ReLU(x)=max(0,x) (функция активации ReLU)
        new_feature_matrix[:,current_col+5]=np.arctan(x);           #6. arctg(x) (арктангенс числа x)        
        new_feature_matrix[:,current_col+6]=np.sinh(x);             #7. sh(x)=(exp(x)-exp(-x))/2 (гиперболический синус)        
        new_feature_matrix[:,current_col+7]=np.cosh(x);             #8. ch(x)=(exp(x)+exp(-x))/2 (гиперболический косинус)        
        new_feature_matrix[:,current_col+8]=np.tanh(x);             #9. th(x)=(exp(2x)-1)/(exp(2x)+1) (гиперболический тангенс)        
        new_feature_matrix[:,current_col+9]=1.0/np.cosh(x);         #10. sch(x)=2/(exp(x)+exp(-x)) (гиперболический секанс)        
        new_feature_matrix[:,current_col+10]=np.log(np.abs(x)+1e-9);#11. ln_abs(x)=ln(abs(x)+1e-9) (натуральный логарифм от модуля)
        new_feature_matrix[:,current_col+11]=np.sqrt(np.abs(x));    #12. sqrt_abs(x)=sqrt(abs(x)) (корень из модуля)
        new_feature_matrix[:,current_col+12]=np.exp(-x**2);         #13. gauss(x)=exp(-x^2)
        current_col=current_col+13;#Увеличение current_col на количество добавленных признаков
    return new_feature_matrix;

def load_data_from_npy(rewrite_features_csv_files:bool=False)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Загрузка данных из *.npy файлов (эти файлы содержат массивы NumPy)"""
    assert os.path.exists(path='hw_final_open_data.npy'),'Please, download hw_final_open_data.npy and place it in the working directory'
    assert os.path.exists(path='hw_final_open_target.npy'),'Please, download hw_final_open_target.npy and place it in the working directory'
    assert os.path.exists(path='hw_final_closed_data.npy'),'Please, download hw_final_closed_data.npy and place it in the working directory'
    opened_data_all_features:np.ndarray=np.load(file='hw_final_open_data.npy',allow_pickle=False);#Загрузка данных
    closed_data_all_features:np.ndarray=np.load(file='hw_final_closed_data.npy',allow_pickle=False);
    opened_target:np.ndarray=np.load(file='hw_final_open_target.npy',allow_pickle=False);
    closed_target:np.ndarray=np.ndarray(shape=(closed_data_all_features.shape[0],),dtype=opened_target.dtype);
    n_samples_opened:int=opened_data_all_features.shape[0];
    n_features_opened:int=opened_data_all_features.shape[1];
    n_samples_closed:int=closed_data_all_features.shape[0];
    n_features_closed:int=closed_data_all_features.shape[1];
    print(f'ПЕРЕД ТРАНСФОРМАЦИЯМИ ПРИЗНАКОВ:');
    print(f'n_samples_opened: {n_samples_opened}, n_features_opened: {n_features_opened}');
    print(f'n_samples_closed: {n_samples_closed}, n_features_closed: {n_features_closed}');#Изначально 7 признаков
    print(f'type(opened_data_all_features): {type(opened_data_all_features)}, type(closed_data_all_features): {type(closed_data_all_features)}');
    print(f'opened_data_all_features.shape: {opened_data_all_features.shape}, closed_data_all_features.shape: {closed_data_all_features.shape}');
    print(f'opened_data_all_features[0]: {opened_data_all_features[0]}');
    print(f'opened_data_all_features[1]: {opened_data_all_features[1]}');
    print(f'opened_data_all_features[2]: {opened_data_all_features[2]}');
    print(f'opened_data_all_features[{n_samples_opened-1}]: {opened_data_all_features[n_samples_opened-1]}');
    print(f'closed_data_all_features[0]: {closed_data_all_features[0]}');
    print(f'closed_data_all_features[1]: {closed_data_all_features[1]}');
    print(f'closed_data_all_features[2]: {closed_data_all_features[2]}');
    print(f'closed_data_all_features[{n_samples_closed-1}]: {closed_data_all_features[n_samples_closed-1]}');
    print(f'type(opened_target): {type(opened_target)}');
    print(f'opened_target.shape: {opened_target.shape}');
    print(f'opened_target[0]: {opened_target[0]}');
    print(f'opened_target[1]: {opened_target[1]}');
    print(f'opened_target[2]: {opened_target[2]}');
    print(f'opened_target[{n_samples_opened-1}]: {opened_target[n_samples_opened-1]}');
    print(f'ПОСЛЕ ТРАНСФОРМАЦИИ ПРИЗНАКОВ data_transformation_pairwise_multiplications:');
    opened_data_all_features=data_transformation_pairwise_multiplications(feature_matrix=opened_data_all_features);
    closed_data_all_features=data_transformation_pairwise_multiplications(feature_matrix=closed_data_all_features);
    n_samples_opened:int=opened_data_all_features.shape[0];
    n_features_opened:int=opened_data_all_features.shape[1];
    n_samples_closed:int=closed_data_all_features.shape[0];
    n_features_closed:int=closed_data_all_features.shape[1];
    print(f'n_samples_opened: {n_samples_opened}, n_features_opened: {n_features_opened}');
    print(f'n_samples_closed: {n_samples_closed}, n_features_closed: {n_features_closed}');#После попарных произведений 7+7*6/2=28 признаков
    print(f'type(opened_data_all_features): {type(opened_data_all_features)}, type(closed_data_all_features): {type(closed_data_all_features)}');
    print(f'opened_data_all_features.shape: {opened_data_all_features.shape}, closed_data_all_features.shape: {closed_data_all_features.shape}');
    print(f'opened_data_all_features[0]: {opened_data_all_features[0]}');
    print(f'opened_data_all_features[1]: {opened_data_all_features[1]}');
    print(f'opened_data_all_features[2]: {opened_data_all_features[2]}');
    print(f'opened_data_all_features[{n_samples_opened-1}]: {opened_data_all_features[n_samples_opened-1]}');
    print(f'closed_data_all_features[0]: {closed_data_all_features[0]}');
    print(f'closed_data_all_features[1]: {closed_data_all_features[1]}');
    print(f'closed_data_all_features[2]: {closed_data_all_features[2]}');
    print(f'closed_data_all_features[{n_samples_closed-1}]: {closed_data_all_features[n_samples_closed-1]}');
    print(f'type(opened_target): {type(opened_target)}');
    print(f'opened_target.shape: {opened_target.shape}');
    print(f'opened_target[0]: {opened_target[0]}');
    print(f'opened_target[1]: {opened_target[1]}');
    print(f'opened_target[2]: {opened_target[2]}');
    print(f'opened_target[{n_samples_opened-1}]: {opened_target[n_samples_opened-1]}');
    print(f'ПОСЛЕ ТРАНСФОРМАЦИИ ПРИЗНАКОВ data_transformation_add_functions:');
    opened_data_all_features=data_transformation_add_functions(feature_matrix=opened_data_all_features);
    closed_data_all_features=data_transformation_add_functions(feature_matrix=closed_data_all_features);
    n_samples_opened:int=opened_data_all_features.shape[0];
    n_features_opened:int=opened_data_all_features.shape[1];
    n_samples_closed:int=closed_data_all_features.shape[0];
    n_features_closed:int=closed_data_all_features.shape[1];
    print(f'n_samples_opened: {n_samples_opened}, n_features_opened: {n_features_opened}');
    print(f'n_samples_closed: {n_samples_closed}, n_features_closed: {n_features_closed}');#После добавления функций 28*(1+13)=392 признаков
    print(f'type(opened_data_all_features): {type(opened_data_all_features)}, type(closed_data_all_features): {type(closed_data_all_features)}');
    print(f'opened_data_all_features.shape: {opened_data_all_features.shape}, closed_data_all_features.shape: {closed_data_all_features.shape}');
    print(f'opened_data_all_features[0]: {opened_data_all_features[0]}');
    print(f'opened_data_all_features[1]: {opened_data_all_features[1]}');
    print(f'opened_data_all_features[2]: {opened_data_all_features[2]}');
    print(f'opened_data_all_features[{n_samples_opened-1}]: {opened_data_all_features[n_samples_opened-1]}');
    print(f'closed_data_all_features[0]: {closed_data_all_features[0]}');
    print(f'closed_data_all_features[1]: {closed_data_all_features[1]}');
    print(f'closed_data_all_features[2]: {closed_data_all_features[2]}');
    print(f'closed_data_all_features[{n_samples_closed-1}]: {closed_data_all_features[n_samples_closed-1]}');
    print(f'type(opened_target): {type(opened_target)}');
    print(f'opened_target.shape: {opened_target.shape}');
    print(f'opened_target[0]: {opened_target[0]}');
    print(f'opened_target[1]: {opened_target[1]}');
    print(f'opened_target[2]: {opened_target[2]}');
    print(f'opened_target[{n_samples_opened-1}]: {opened_target[n_samples_opened-1]}');

    opened_ids:np.ndarray=np.ndarray(shape=(opened_data_all_features.shape[0],),dtype='U4');
    for i in range(opened_data_all_features.shape[0]):
        opened_ids[i]=f'{i:04}';
        #print(f'i: {i}, opened_ids[i]: {opened_ids[i]}');
    print(f'opened_ids[0]: {opened_ids[0]}, opened_ids[799]: {opened_ids[799]}');
    print(f'opened_ids.shape: {opened_ids.shape}');

    closed_ids:np.ndarray=np.ndarray(shape=(closed_data_all_features.shape[0],),dtype='U4');
    for i in range(closed_data_all_features.shape[0]):
        closed_ids[i]=f'{i:04}';
        #print(f'i: {i}, opened_ids[i]: {opened_ids[i]}');
    print(f'closed_ids[0]: {closed_ids[0]}, closed_ids[199]: {closed_ids[199]}');
    print(f'closed_ids.shape: {closed_ids.shape}');

    if rewrite_features_csv_files==True:#Перезапись csv файлов со значениями всех признаков для открытых и закрытых данных
        opened_data_all_features_csv_file_name:str=f'all_features_opened_data.csv';
        closed_data_all_features_csv_file_name:str=f'all_features_closed_data.csv';
        opened_buf_str_list:list[str]=[];
        closed_buf_str_list:list[str]=[];
        buf_s:str='id,'+','.join(['X'+str(i)for i in range(n_features_opened)])+',Y'+'\n';#Добавление заголовков csv файлов
        opened_buf_str_list.append(buf_s);
        buf_s:str='id,'+','.join(['X'+str(i)for i in range(n_features_closed)])+',Y'+'\n';
        closed_buf_str_list.append(buf_s);
        for sample_index in range(n_samples_opened):
            buf_s:str=opened_ids[sample_index]+','+','.join([str(float(opened_data_all_features[sample_index][feature_index]))for feature_index in range(n_features_opened)])+','+str(float(opened_target[sample_index]))+'\n';
            opened_buf_str_list.append(buf_s);
        for sample_index in range(n_samples_closed):
            buf_s:str=closed_ids[sample_index]+','+','.join([str(float(closed_data_all_features[sample_index][feature_index]))for feature_index in range(n_features_closed)])+','+str(float(closed_target[sample_index]))+'\n';
            closed_buf_str_list.append(buf_s);
        with open(file=opened_data_all_features_csv_file_name,mode='wt',encoding='UTF-8')as f_csv:f_csv.writelines(opened_buf_str_list);
        with open(file=closed_data_all_features_csv_file_name,mode='wt',encoding='UTF-8')as f_csv:f_csv.writelines(closed_buf_str_list);
    return opened_data_all_features,opened_target,opened_ids,closed_data_all_features,closed_target,closed_ids;

def create_log_files()->None:
    """Функция создаёт log файлы (если они не существуют)"""
    if pathlib.Path('log_pipelines.txt').exists()==False:#Создать файл log_pipelines.txt если его не существует
        with open(file='log_pipelines.txt',mode='wt',encoding='UTF-8')as f_log:pass;
    if pathlib.Path('log_pipelines.csv').exists()==False:#Создать файл log_pipelines.csv если его не существует и заполнить его заголовок
        with open(file='log_pipelines.csv',mode='wt',encoding='UTF-8')as f_log:
            header_str:str=f'pipeline_id,n_features_all,n_features_selected_randomly,use_imputer,imputer_type,use_var_thresholder,var_thresholder_type,use_scaler,scaler_type,use_feature_selector,feature_selector_type,fs_score_func_type,fs_estimator_type,model_type,score_type,score_valid_mean,score_valid_std,score_test,dt_pipe_start_str,seconds_processing,pipeline_file_size';
            print(header_str,file=f_log);
    if pathlib.Path('log_results.txt').exists()==False:#Создать файл log_results.txt если его не существует
        with open(file='log_results.txt',mode='wt',encoding='UTF-8')as f_log:pass;

def run_one_pipeline_experiment_v1(num_features_select_from_all_min:int=5,num_features_select_from_all_max:int=50,randomly_selected_indexes:list[int]=None,problem_type:str='regression',task_output:str='mono_output',score_type:str='mean_squared_error',fbeta_score_beta:float=1.0,d2_pinball_score_alpha:float=0.5,d2_tweedie_score_power:float=0.0,mean_pinball_loss_alpha:float=0.5,mean_tweedie_deviance_power:float=0.0,use_imputer_probability:float=0.95,imputer_type:str=None,imputer_hyperparams:dict=None,use_var_thresholder_probability:float=0.95,var_thresholder_type:str=None,var_thresholder_hyperparams:dict=None,use_feature_selector_probability:float=0.7,feature_selector_type:str=None,prefered_feature_selector_types:list[str]=None,feature_selector_hyperparams:dict=None,fs_score_func_type:str=None,fs_estimator_type:str=None,prefered_fs_estimator_types:list[str]=None,fs_estimator_hyperparams:dict=None,use_scaler_probability:float=0.9,scaler_type:str=None,prefered_scaler_types:list[str]=None,scaler_hyperparams:dict=None,model_type:str=None,prefered_model_types:list[str]=None,model_hyperparams:dict=None,num_folds:int=10,score_valid_min_threshold:float=None,score_valid_max_threshold:float=None,non_negative_y_guarantee:bool=False,use_only_linear_models:bool=False,n_cpu_cores:int=-1)->str:
    """
    Запуск одного эксперимента со случайным выбором пайплайна, его компонентов и их гиперпараметров\n
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
    r2_score,root_mean_squared_error,root_mean_squared_log_error\n\n\n

    Сначала из матриц признаков opened_data_all_features и closed_data_all_features выбираются
    некоторым способом num_features_select_from_all столбцов (признаков), из которых составляются
    матрицы opened_data и closed_data соответственно\n
    Если num_features_select_from_all=0, то используются все признаки.
    Значение num_features_select_from_all определяется через num_features_select_from_all_min
    и num_features_select_from_all_max.\n
    Если в качестве параметра randomly_selected_indexes передаётся список, то отбираются именно эти признаки, а не случайные.\n

    use_only_linear_models:bool=False - если True, то выбираются только те пайплайны, у моделей которых есть атрибуты 'coef_' и 'intercept_'\n

    use_scaler_probability:float=0.9,scaler_type:str=None,scaler_hyperparams:dict=None - эти параметры позволяют использовать
    различные варианты scaler (или не использовать scaler вообще если use_scaler=False)
        
    fs_score_func_type - это тип функции, которая используется для feature_selector, если он равен GenericUnivariateSelect,SelectFdr,SelectFpr,
    SelectFwe,SelectKBest,SelectPercentile. Возможные значения: chi2,f_classif,f_regression,mutual_info_classif,mutual_info_regression,
    r_regression.

    Пайплайн: imputer -> var_thresholder -> scaler -> feature_selector -> model
    """
    dt_pipe_start:datetime.datetime=datetime.datetime.now();#Для лога (когда этот пайплайн запущен)
    dt_pipe_start_str:str=dt_pipe_start.strftime(format='%Y-%m-%d_%H-%M-%S');
    seconds_pipe_start:float=time.time();#Для лога (чтобы вычислить время обработки этого пайплайна)
    print(f'Функция run_one_pipeline_experiment_v1 вызвана с параметрами: {locals()}');
    error_str:str='PIPELINE_ERROR';
    use_imputer:bool=true_with_prob(p=use_imputer_probability);#Использовать ли imputer в этом эксперименте
    use_var_thresholder:bool=true_with_prob(p=use_var_thresholder_probability);#Использовать ли var_thresholder в этом эксперименте
    use_scaler:bool=true_with_prob(p=use_scaler_probability);#Использовать ли scaler в этом эксперименте
    use_feature_selector:bool=true_with_prob(p=use_feature_selector_probability);#Использовать ли feature_selector в этом эксперименте
    # 1. Загрузка ВСЕХ данных (открытых и закрытых) и отбор num_features_select_from_all признаков из всех
    #Загрузка выполняется отдельно, так как:
    #1) Если эксперимент повторяется много раз, загружать данные каждый раз неэффективно по времени
    #2) Данные могут быть представлены в разных форматах (csv,json,npy,...), поэтому обработку каждого из этих форматов лучше
    #выполнять отдельно в своей функции (load_data_from_npy, load_data_from_csv, load_data_from_json, ...)
    #Отбор num_features_select_from_all признаков:
    n_samples_opened:int=opened_data_all_features.shape[0];
    n_samples_closed:int=closed_data_all_features.shape[0];
    n_features_all:int=opened_data_all_features.shape[1];#Равно opened_data_all_features.shape[1]    
    if randomly_selected_indexes is None:#Если индексы не заданы, то они выбираются случайным образом
        #ЕСЛИ ИНДЕКСЫ ВЫБИРАЮТСЯ СЛУЧАЙНЫМ ОБРАЗОМ, ТО НИКАКАЯ ИНФОРМАЦИЯ О ТЕСТОВОЙ ВЫБОРКЕ НЕ ИСПОЛЬЗУЕТСЯ => УТЕЧКИ ДАННЫХ НЕТ
        if num_features_select_from_all_min==num_features_select_from_all_max==0:num_features_select_from_all=0;
        else:num_features_select_from_all:int=random.randint(a=num_features_select_from_all_min,b=num_features_select_from_all_max);
        n_features_selected_randomly:int=num_features_select_from_all;
        if num_features_select_from_all==0:#Если num_features_select_from_all=0 и индексы не заданы, то используются все признаки
            all_indexes:list[int]=[i for i in range(n_features_all)];#Все индексы 0..n_features_all-1
            #ЕСЛИ ВЫБИРАЮТСЯ ВСЕ ИНДЕКСЫ, ТО НИКАКАЯ ИНФОРМАЦИЯ О ТЕСТОВОЙ ВЫБОРКЕ НЕ ИСПОЛЬЗУЕТСЯ => УТЕЧКИ ДАННЫХ НЕТ
            randomly_selected_indexes:list[int]=[i for i in range(n_features_all)];#Все индексы 0..n_features_all-1
        else:#Если num_features_select_from_all!=0 и индексы не заданы, то выполняется отбор случайного множества признаков из всех        
            all_indexes:list[int]=[i for i in range(n_features_all)];#Все индексы 0..n_features_all-1
            for i in range(10):random.shuffle(x=all_indexes);#Shuffle list x in place, and return None.
            randomly_selected_indexes:list[int]=[all_indexes[i] for i in range(num_features_select_from_all)];#Индексы выбранных признаков
            #(то есть первые num_features_select_from_all индексов из всех n_features_all индексов, перемешанных случайным образом)
            randomly_selected_indexes.sort();#Эта сортировка по идее ни на что не влияет, но так просто удобнее для человека
            #Использование списка randomly_selected_indexes и его сохранение в лог делает эсперименты воспроизводимыми
    else:#Если индексы заданы, то выбираются столбцы именно с этими индексами (ничего случайным образом не выбирается, ВНЕ зависимости
        #от num_features_select_from_all_min и num_features_select_from_all_max)
        n_features_selected_randomly:int=len(randomly_selected_indexes);
        #ЕСЛИ ИНДЕКСТЫ ЗАДАНЫ, ТО ЭТО ИСПОЛЬЗУЕТСЯ ДЛЯ ВОСПРОИЗВОДИМОСТИ ЭКСПЕРИМЕНТОВ, ПОЭТОМУ УТЕЧКИ ДАННЫХ ТОЖЕ НЕТ
    #Для создания и заполнения массивов opened_data и closed_data используется один и тот же код вне зависимости от переданных
    #значений randomly_selected_indexes, num_features_select_from_all_min и num_features_select_from_all_max
    opened_data:np.ndarray=np.zeros((n_samples_opened,n_features_selected_randomly));#Создание новых массивов
    closed_data:np.ndarray=np.zeros((n_samples_closed,n_features_selected_randomly));
    for col_idx,feature_idx in enumerate(randomly_selected_indexes):#Копирование столбцов с выбранными индексами из исходных массивов
        opened_data[:,col_idx]=opened_data_all_features[:,feature_idx];
        closed_data[:,col_idx]=closed_data_all_features[:,feature_idx];
    #Все этапы пайплайна (imputer, var_thresholder, scaler, feature_selector, model) применяются к одному и тому же набору признаков (который
    #выбран в соответствии со списком randomly_selected_indexes)

    # 2. Разделение на train (для cross_valid) и final_test (для оценки на holdout)
    split_random_state:int=int(time.time()*(10**9))%(2**32);#Количество наносекунд с начала эпохи Unix -> [0, 4294967295]
    hyperparam_random_state:int=int(random.uniform(a=0.0,b=1e20))%(2**32);
    score_func_random_state:int=int(random.uniform(a=0.0,b=1e20))%(2**32);
    opened_data_len:int=opened_data.shape[0];
    if problem_type=='classification':
        X_train_cv,X_test_final,y_train_cv,y_test_final,ids_train_cv,ids_test_final=train_test_split(
        opened_data,opened_target,opened_ids,test_size=0.2,random_state=split_random_state,stratify=opened_target);
    elif problem_type=='regression':
        X_train_cv,X_test_final,y_train_cv,y_test_final,ids_train_cv,ids_test_final=train_test_split(
        opened_data,opened_target,opened_ids,test_size=0.2,random_state=split_random_state);
    X_train_cv:np.ndarray;X_test_final:np.ndarray;y_train_cv:np.ndarray;y_test_final:np.ndarray;ids_train_cv:np.ndarray;ids_test_final:np.ndarray;
    # Проверяем размеры
    #print(f"Размер train для CV: {X_train_cv.shape}")
    #print(f"Размер final test: {X_test_final.shape}")
    print(f'Доля X_train_cv от opened_data: {X_train_cv.shape[0]/opened_data_len}');#0.8
    print(f'Доля X_test_final от opened_data: {X_test_final.shape[0]/opened_data_len}');#0.2

    #3.1. Установка типа и гиперпараметров imputer
    if use_imputer==True:
        if imputer_type is None:#Выбор случайного imputer_type из списка:
            imputer_types:list[str]=['KNNImputer','SimpleImputer'];
            imputer_type=random.choice(seq=imputer_types);
        if imputer_hyperparams is None:
            if imputer_type=='KNNImputer':imputer_hyperparams:dict={'n_neighbors':random.randint(a=1,b=15),'weights':random.choice(seq=['uniform','distance']),'metric':'nan_euclidean','copy':True,'add_indicator':random.choice(seq=[True,False]),'keep_empty_features':random.choice(seq=[True,False])};
            elif imputer_type=='SimpleImputer':imputer_hyperparams:dict={'strategy':random.choice(seq=['mean','median','most_frequent','constant']),'fill_value':None,'copy':True,'add_indicator':random.choice(seq=[True,False]),'keep_empty_features':random.choice(seq=[True,False])};
    else:
        print(f'Выбрано use_imputer==False, imputer не применяется');
        imputer_hyperparams=None;imputer_type=None;

    #3.2. Установка типа и гиперпараметров var_thresholder
    if use_var_thresholder==True:
        if var_thresholder_type is None:#Выбор случайного var_thresholder_type из списка:
            var_thresholder_types:list[str]=['VarianceThreshold'];
            var_thresholder_type=random.choice(seq=var_thresholder_types);
        if var_thresholder_hyperparams is None:
            if var_thresholder_type=='VarianceThreshold':var_thresholder_hyperparams:dict={'threshold':10**random.uniform(a=-15,b=1)};
    else:
        print(f'Выбрано use_var_thresholder==False, var_thresholder не применяется');
        var_thresholder_hyperparams=None;var_thresholder_type=None;

    #3.3. Установка типа и гиперпараметров scaler
    if use_scaler==True:
        if scaler_type is None:#Выбор случайного scaler из списка:
            scaler_types:list[str]=['MaxAbsScaler','MinMaxScaler','RobustScaler','StandardScaler'];
            if prefered_scaler_types is not None:scaler_types=list(set(scaler_types).intersection(set(prefered_scaler_types)));
            scaler_type:str=random.choice(seq=scaler_types);
        if scaler_hyperparams is None:
            if scaler_type=='MaxAbsScaler':scaler_hyperparams={'copy':True};
            elif scaler_type=='MinMaxScaler':scaler_hyperparams={'feature_range':(0,1),'copy':True,'clip':False};
            elif scaler_type=='RobustScaler':scaler_hyperparams={'with_centering':true_with_prob(p=0.85),'with_scaling':true_with_prob(p=0.85),'quantile_range':(random.uniform(a=0.0,b=40.0),random.uniform(a=60.0,b=100.0)),'copy':True,'unit_variance':true_with_prob(p=0.1)};
            elif scaler_type=='StandardScaler':scaler_hyperparams={'copy':True,'with_mean':true_with_prob(p=0.85),'with_std':true_with_prob(p=0.85)};
    else:
        print(f'Выбрано use_scaler==False, scaler не применяется');
        scaler_hyperparams=None;scaler_type=None;

    #3.4. Установка типа и гиперпараметров feature_selector
    #Фактически все feature_selector делятся на 2 вида:
    #1) которые используют score_func (значимость признака оценивается по значению этой функции между этим признаком и target)
    #2) которые используют estimator (значимость признака оценивается по влиянию этого признака на предсказания модели)
    #3 [условно]) VarianceThreshold (условно, так как он используется отдельно перед scaler)
    if use_feature_selector==True:
        if feature_selector_type is None:#Выбор случайного feature_selector из списка:
            feature_selector_types_all:list[str]=['GenericUnivariateSelect','RFE','RFECV','SelectFdr','SelectFpr','SelectFromModel','SelectFwe','SelectKBest','SelectPercentile','SequentialFeatureSelector'];
            feature_selector_types_estimator_rfe:list[str]=['RFE','RFECV'];
            feature_selector_types_estimator_not_rfe:list[str]=['SelectFromModel','SequentialFeatureSelector'];
            feature_selector_types_estimator_all:list[str]=feature_selector_types_estimator_rfe+feature_selector_types_estimator_not_rfe;
            feature_selector_types_score_func_based:list[str]=['GenericUnivariateSelect','SelectFdr','SelectFpr','SelectFwe','SelectKBest','SelectPercentile'];
            if prefered_feature_selector_types is not None:feature_selector_types_all=list(set(feature_selector_types_all).intersection(set(prefered_feature_selector_types)));
            feature_selector_type:str=random.choice(seq=feature_selector_types_all);
        if feature_selector_hyperparams is None:
            if feature_selector_type in feature_selector_types_estimator_rfe:#Нужно выбрать тип и гиперпараметры для estimator
                if problem_type=='classification':fs_estimator_types:list[str]=['LinearSVC','NuSVC','SVC'];
                elif problem_type=='regression':fs_estimator_types:list[str]=['LinearSVR','NuSVR','SVR'];
            elif feature_selector_type in feature_selector_types_estimator_not_rfe:#Нужно выбрать тип и гиперпараметры для estimator
                if problem_type=='classification':
                    fs_estimator_types:list[str]=['LogisticRegression','PassiveAggressiveClassifier','Perceptron','RidgeClassifier','SGDClassifier'];
                elif problem_type=='regression':
                    fs_estimator_types:list[str]=['LinearRegression','Ridge','SGDRegressor','ElasticNet','Lars','Lasso','LassoLars','LassoLarsIC','OrthogonalMatchingPursuit','ARDRegression','BayesianRidge','HuberRegressor','QuantileRegressor','RANSACRegressor','TheilSenRegressor','GammaRegressor','PoissonRegressor','TweedieRegressor','PassiveAggressiveRegressor'];
            if feature_selector_type in feature_selector_types_estimator_all:#Если есть estimator, то нужно выбрать для него гиперпараметры
                if fs_estimator_type is None:
                    if prefered_fs_estimator_types is not None:fs_estimator_types=list(set(fs_estimator_types).intersection(set(prefered_fs_estimator_types)));
                    fs_estimator_type:str=random.choice(seq=fs_estimator_types);
                    if fs_estimator_hyperparams is None:
                        if fs_estimator_type=='LogisticRegression':fs_estimator_hyperparams:dict={'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'dual':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-8,b=0),'C':random.uniform(a=0.5,b=1.5),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']),'max_iter':random.randint(a=50,b=500),'warm_start':random.choice(seq=[True,False]),'l1_ratio':random.uniform(a=0.0,b=1.0),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
                        elif fs_estimator_type=='PassiveAggressiveClassifier':fs_estimator_hyperparams:dict={'C':random.uniform(a=0.5,b=1.5),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=3000),'tol':10**random.uniform(a=-7,b=1),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'shuffle':random.choice(seq=[True,False]),'loss':random.choice(seq=['hinge','squared_hinge']),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[True,False,1,2,3,4,5,6,7,8,9,10]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
                        elif fs_estimator_type=='Perceptron':fs_estimator_hyperparams:dict={'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-12,b=0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-6,b=0),'shuffle':random.choice(seq=[True,False]),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'warm_start':random.choice(seq=[True,False]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
                        elif fs_estimator_type=='RidgeClassifier':fs_estimator_hyperparams:dict={'alpha':10**random.uniform(a=-2,b=1),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-8,b=0),'solver':random.choice(seq=['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']),'positive':random.choice(seq=[True,False]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
                        elif fs_estimator_type=='SGDClassifier':fs_estimator_hyperparams:dict={'loss':random.choice(seq=['hinge','log_loss','modified_huber','squared_hinge','perceptron','squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']),'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-8,b=0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-6,b=0),'shuffle':random.choice(seq=[True,False]),'epsilon':10**random.uniform(a=-9,b=3),'learning_rate':random.choice(seq=['constant','optimal','invscaling','adaptive']),'eta0':10**random.uniform(a=-5,b=1),'power_t':random.uniform(a=-5,b=6),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[True,False,1,2,3,4,5,6,7,8,9,10]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
                        elif fs_estimator_type=='LinearRegression':fs_estimator_hyperparams:dict={'fit_intercept':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-9,b=-3),'positive':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='Ridge':fs_estimator_hyperparams:dict={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=500,b=20000),'tol':10**random.uniform(a=-9,b=-0.1),'solver':random.choice(seq=['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']),'positive':bool(random.randint(a=0,b=1)),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='SGDRegressor':fs_estimator_hyperparams:dict={'loss':random.choice(seq=['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']),'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-9,b=1.0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=20000),'tol':10**random.uniform(a=-6,b=0),'epsilon':10**random.uniform(a=-3,b=1),'learning_rate':random.choice(seq=['constant','optimal','invscaling','adaptive']),'eta0':10**random.uniform(a=-5,b=0),'power_t':random.uniform(a=-100,b=100),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0,b=1),'n_iter_no_change':random.randint(a=2,b=10),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[False,False,False,False,False,False,False,False,False,False,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='ElasticNet':fs_estimator_hyperparams:dict={'alpha':10**random.uniform(a=-5,b=2),'l1_ratio':random.uniform(a=0,b=1),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-7,b=-1),'warm_start':random.choice(seq=[True,False]),'positive':random.choice(seq=[True,False]),'selection':random.choice(seq=['cyclic','random']),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='Lars':fs_estimator_hyperparams:dict={'fit_intercept':random.choice(seq=[True,False]),'n_nonzero_coefs':random.randint(a=10,b=100),'eps':10**random.uniform(a=-5,b=-1),'fit_path':random.choice(seq=[True,False]),'jitter':10**random.uniform(a=-9,b=-1),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='Lasso':fs_estimator_hyperparams:dict={'alpha':10**random.uniform(a=-5,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(100,2000),'tol':10**random.uniform(a=-8,b=-1),'warm_start':random.choice(seq=[True,False]),'positive':random.choice(seq=[True,False]),'selection':random.choice(seq=['cyclic','random']),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='LassoLars':fs_estimator_hyperparams:dict={'alpha':10**random.uniform(a=-5,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=1500),'eps':10**random.uniform(a=-10,b=-5),'fit_path':random.choice(seq=[True,False]),'positive':random.choice(seq=[True,False]),'jitter':10**random.uniform(a=-9,b=-1),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='LassoLarsIC':fs_estimator_hyperparams:dict={'criterion':random.choice(seq=['aic','bic']),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=1500),'eps':10**random.uniform(a=-16,b=-10),'positive':random.choice(seq=[True,False]),'noise_variance':10**random.uniform(a=-5,b=-1)};
                        elif fs_estimator_type=='OrthogonalMatchingPursuit':fs_estimator_hyperparams:dict={'n_nonzero_coefs':None if true_with_prob(p=0.5)==True else random.randint(a=5,b=50),'tol':None if true_with_prob(p=0.5)==True else 10**random.uniform(a=-2,b=2),'fit_intercept':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='ARDRegression':fs_estimator_hyperparams:dict={'max_iter':random.randint(a=100,b=700),'tol':10**random.uniform(a=-7,b=-1),'alpha_1':10**random.uniform(a=-10,b=-2),'alpha_2':10**random.uniform(a=-10,b=-2),'lambda_1':10**random.uniform(a=-10,b=-2),'lambda_2':10**random.uniform(a=-10,b=-2),'compute_score':random.choice(seq=[True,False]),'threshold_lambda':random.uniform(a=5000,b=15000),'fit_intercept':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='BayesianRidge':fs_estimator_hyperparams:dict={'max_iter':random.randint(a=100,b=700),'tol':10**random.uniform(a=-5,b=-1),'alpha_1':10**random.uniform(a=-10,b=-2),'alpha_2':10**random.uniform(a=-10,b=-2),'lambda_1':10**random.uniform(a=-10,b=-2),'lambda_2':10**random.uniform(a=-10,b=-2),'alpha_init':random.uniform(a=0.01,b=1.0),'lambda_init':random.uniform(a=0.01,b=1.0),'compute_score':random.choice(seq=[True,False]),'fit_intercept':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='HuberRegressor':fs_estimator_hyperparams:dict={'epsilon':random.uniform(a=1.0,b=10.0),'max_iter':random.randint(a=20,b=200),'alpha':10**random.uniform(a=-8,b=0),'warm_start':random.choice(seq=[True,False]),'fit_intercept':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-8,b=-2)};
                        elif fs_estimator_type=='QuantileRegressor':
                            fs_estimator_hyperparams:dict={'quantile':random.uniform(a=0.0,b=1.0),'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['highs-ds','highs-ipm','highs','interior-point','revised simplex'])};
                            #Solver interior-point is not anymore available in SciPy >= 1.11.0
                            if fs_estimator_hyperparams['solver']=='interior-point':fs_estimator_hyperparams['solver']=random.choice(seq=['highs-ds','highs-ipm','highs','revised simplex']);
                        elif fs_estimator_type=='RANSACRegressor':fs_estimator_hyperparams:dict={'min_samples':random.uniform(a=0.0,b=1.0),'max_trials':random.randint(a=50,b=150),'max_skips':random.randint(a=500,b=1000),'stop_n_inliers':random.randint(a=500,b=1000),'stop_score':10**random.uniform(a=3,b=10),'stop_probability':random.uniform(a=0.95,b=1.00),'loss':random.choice(seq=['absolute_error','squared_error']),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='TheilSenRegressor':fs_estimator_hyperparams:dict={'fit_intercept':random.choice(seq=[True,False]),'max_subpopulation':10**random.uniform(a=-6,b=-2),'n_subsamples':random.randint(a=opened_data.shape[1]+1,b=opened_data.shape[0]),'max_iter':random.randint(a=100,b=500),'tol':10**random.uniform(a=-5,b=-1),'n_jobs':n_cpu_cores,'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='GammaRegressor':fs_estimator_hyperparams:dict={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-9,b=-2),'warm_start':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='PoissonRegressor':fs_estimator_hyperparams:dict={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-6,b=-2),'warm_start':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='TweedieRegressor':fs_estimator_hyperparams:dict={'power':random.choice(seq=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.01,1.02,1.03,1.04,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,1.96,1.97,1.98,1.99,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0]),'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'link':random.choice(seq=['auto','identity','log']),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-6,b=-2),'warm_start':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='PassiveAggressiveRegressor':fs_estimator_hyperparams:dict={'C':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=2000),'tol':10**random.uniform(a=-5,b=-1),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=0.9),'n_iter_no_change':random.randint(a=2,b=10),'loss':random.choice(seq=['epsilon_insensitive','squared_epsilon_insensitive']),'epsilon':random.uniform(a=0.05,b=0.15),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[False,False,False,False,False,False,False,False,False,False,False,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='LinearSVC':fs_estimator_hyperparams:dict={'penalty':random.choice(seq=['l1','l2']),'loss':random.choice(seq=['hinge','squared_hinge']),'dual':random.choice(seq=['auto',True,False]),'tol':10**random.uniform(a=-6,b=-2),'C':10**random.uniform(a=-2,b=2),'fit_intercept':random.choice(seq=[True,False]),'intercept_scaling':10**random.uniform(a=-2,b=2),'random_state':hyperparam_random_state,'max_iter':random.randint(a=500,b=5000)};
                        elif fs_estimator_type=='NuSVC':fs_estimator_hyperparams:dict={'nu':random.uniform(a=0.0,b=1.0),'kernel':random.choice(seq=['linear','poly','rbf','sigmoid','precomputed']),'degree':random.randint(a=0,b=5),'gamma':random.choice(seq=['auto','scale',random.uniform(a=0.0,b=0.1)]),'shrinking':random.choice(seq=[True,False]),'probability':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-4,b=-2),'class_weight':random.choice(seq=['balanced',None]),'max_iter':random.randint(a=500,b=2000),'decision_function_shape':random.choice(seq=['ovo','ovr']),'break_ties':random.choice(seq=[True,False]),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='SVC':fs_estimator_hyperparams:dict={'C':10**random.uniform(a=-2,b=2),'kernel':random.choice(seq=['linear','poly','rbf','sigmoid','precomputed']),'degree':random.randint(a=0,b=5),'gamma':random.choice(seq=['auto','scale',random.uniform(a=0.0,b=0.1)]),'shrinking':random.choice(seq=[True,False]),'probability':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-4,b=-2),'class_weight':random.choice(seq=['balanced',None]),'max_iter':random.randint(a=500,b=2000),'decision_function_shape':random.choice(seq=['ovo','ovr']),'break_ties':random.choice(seq=[True,False]),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='LinearSVR':fs_estimator_hyperparams:dict={'epsilon':0.0,'tol':10**random.uniform(a=-6,b=-2),'C':10**random.uniform(a=-2,b=2),'loss':random.choice(seq=['epsilon_insensitive','squared_epsilon_insensitive']),'fit_intercept':random.choice(seq=[True,False]),'intercept_scaling':10**random.uniform(a=-2,b=2),'dual':random.choice(seq=['auto',True,False]),'random_state':hyperparam_random_state,'max_iter':random.randint(a=500,b=5000)};
                        elif fs_estimator_type=='NuSVR':fs_estimator_hyperparams:dict={'nu':random.uniform(a=0.0,b=1.0),'C':10**random.uniform(a=-2,b=2),'kernel':random.choice(seq=['linear','poly','rbf','sigmoid','precomputed']),'degree':random.randint(a=0,b=5),'gamma':random.choice(seq=['auto','scale',random.uniform(a=0.0,b=0.1)]),'shrinking':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-4,b=-2),'max_iter':random.randint(a=500,b=5000)};
                        elif fs_estimator_type=='SVR':fs_estimator_hyperparams:dict={'kernel':random.choice(seq=['linear','poly','rbf','sigmoid','precomputed']),'degree':random.randint(a=0,b=5),'gamma':random.choice(seq=['auto','scale',random.uniform(a=0.0,b=0.1)]),'tol':10**random.uniform(a=-4,b=-2),'C':10**random.uniform(a=-2,b=2),'epsilon':random.uniform(a=0.05,b=0.15),'shrinking':random.choice(seq=[True,False]),'max_iter':random.randint(a=500,b=5000)};

            if feature_selector_type=='GenericUnivariateSelect':
                feature_selector_hyperparams={'mode':random.choice(seq=['percentile','k_best','fpr','fdr','fwe'])};
                if feature_selector_hyperparams['mode']=='percentile':feature_selector_hyperparams['param']=random.choice(seq=[1,2,5,10,25,50,75,80,90,95,98,99]);
                elif feature_selector_hyperparams['mode']=='k_best':feature_selector_hyperparams['param']=random.choice(seq=[5,10,20,30,50,100]);
                elif feature_selector_hyperparams['mode']in['fpr','fdr','fwe']:feature_selector_hyperparams['param']=10.0**random.uniform(a=-6.0,b=-1.0);
                if feature_selector_hyperparams['mode']=='k_best':
                    if feature_selector_hyperparams['param']>n_features_selected_randomly:feature_selector_hyperparams['param']=n_features_selected_randomly;
                if problem_type=='classification':score_func_types:list[str]=['f_classif','mutual_info_classif','chi2'];
                elif problem_type=='regression':score_func_types:list[str]=['f_regression','mutual_info_regression'];
                #Проверка (и исправление при необходимости) совместимости problem_type и fs_score_func_type:
                if problem_type=='classification'and fs_score_func_type in ['f_regression','mutual_info_regression']:
                    fs_score_func_type=random.choice(seq=['f_classif','mutual_info_classif','chi2']);
                if problem_type=='regression'and fs_score_func_type in ['f_classif','mutual_info_classif','chi2']:
                    fs_score_func_type=random.choice(seq=['f_regression','mutual_info_regression']);
                #Выбор заданной пользователем или случайной функции для feature_selector:
                if fs_score_func_type==None:fs_score_func_type:str=random.choice(seq=score_func_types);
                #В итоге переменная score_func_hyperparams не используется, так как в score_func передаётся именно функция, а не результат
                #вызова функции с некоторым набором параметров
                if fs_score_func_type=='f_classif':score_func_hyperparams:dict={};
                elif fs_score_func_type=='mutual_info_classif':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
                elif fs_score_func_type=='chi2':score_func_hyperparams:dict={};
                elif fs_score_func_type=='f_regression':score_func_hyperparams:dict={'center':random.choice(seq=[True,False]),'force_finite':random.choice(seq=[True,False])};
                elif fs_score_func_type=='mutual_info_regression':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
            #Если feature_selector_type - это один из RFE,RFECV,SelectFromModel,SequentialFeatureSelector, то для него нужна отдельная
            #модель (estimator), по которой определяется важность признаков. Пока что все эти случаи закрыты заглушками (то есть просто
            #считаем, что feature_selector не используется)
            elif feature_selector_type=='RFE':#estimator будем использовать уже отдельно для cross_valid, final_test и production
                feature_selector_hyperparams:dict={'n_features_to_select':random.uniform(a=0.1,b=0.9),'step':random.randint(a=1,b=2),'importance_getter':'auto'};
            elif feature_selector_type=='RFECV':#estimator будем использовать уже отдельно для cross_valid, final_test и production
                feature_selector_hyperparams:dict={'step':random.randint(a=1,b=2),'min_features_to_select':random.randint(a=1,b=10),'cv':random.randint(a=3,b=15),'scoring':None,'n_jobs':n_cpu_cores,'importance_getter':'auto'};
            elif feature_selector_type in ['SelectFdr','SelectFpr','SelectFwe']:
                feature_selector_hyperparams={'alpha':10.0**random.uniform(a=-4.0,b=-0.1)};
                if problem_type=='classification':score_func_types:list[str]=['f_classif','mutual_info_classif','chi2'];
                elif problem_type=='regression':score_func_types:list[str]=['f_regression','mutual_info_regression'];
                fs_score_func_type:str=random.choice(seq=score_func_types);
                if fs_score_func_type=='f_classif':score_func_hyperparams:dict={};
                elif fs_score_func_type=='mutual_info_classif':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
                elif fs_score_func_type=='chi2':score_func_hyperparams:dict={};
                elif fs_score_func_type=='f_regression':score_func_hyperparams:dict={'center':random.choice(seq=[True,False]),'force_finite':random.choice(seq=[True,False])};
                elif fs_score_func_type=='mutual_info_regression':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
            elif feature_selector_type=='SelectFromModel':#estimator будем использовать уже отдельно для cross_valid, final_test и production
                feature_selector_hyperparams:dict={'threshold':None,'prefit':False,'norm_order':random.randint(a=1,b=3),'max_features':random.randint(a=50,b=150),'importance_getter':'auto'};
            elif feature_selector_type=='SelectKBest':
                feature_selector_hyperparams={'k':random.randint(a=10,b=100)};
                if feature_selector_hyperparams['k']>n_features_selected_randomly:feature_selector_hyperparams['k']=n_features_selected_randomly;
                if problem_type=='classification':
                    score_func_types:list[str]=['f_classif','mutual_info_classif','chi2'];
                elif problem_type=='regression':
                    score_func_types:list[str]=['f_regression','mutual_info_regression'];
                fs_score_func_type:str=random.choice(seq=score_func_types);
                if fs_score_func_type=='f_classif':score_func_hyperparams:dict={};
                elif fs_score_func_type=='mutual_info_classif':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
                elif fs_score_func_type=='chi2':score_func_hyperparams:dict={};
                elif fs_score_func_type=='f_regression':score_func_hyperparams:dict={'center':random.choice(seq=[True,False]),'force_finite':random.choice(seq=[True,False])};
                elif fs_score_func_type=='mutual_info_regression':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
            elif feature_selector_type=='SelectPercentile':
                feature_selector_hyperparams={'percentile':random.randint(a=10,b=100)};
                if problem_type=='classification':
                    score_func_types:list[str]=['f_classif','mutual_info_classif','chi2'];
                elif problem_type=='regression':
                    score_func_types:list[str]=['f_regression','mutual_info_regression'];
                fs_score_func_type:str=random.choice(seq=score_func_types);
                if fs_score_func_type=='f_classif':score_func_hyperparams:dict={};
                elif fs_score_func_type=='mutual_info_classif':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
                elif fs_score_func_type=='chi2':score_func_hyperparams:dict={};
                elif fs_score_func_type=='f_regression':score_func_hyperparams:dict={'center':random.choice(seq=[True,False]),'force_finite':random.choice(seq=[True,False])};
                elif fs_score_func_type=='mutual_info_regression':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
            elif feature_selector_type=='SequentialFeatureSelector':#estimator будем использовать уже отдельно для cross_valid, final_test и production
                feature_selector_hyperparams:dict={'n_features_to_select':random.uniform(a=0.1,b=0.9),'tol':10**random.uniform(a=-2,b=0),'direction':random.choice(seq=['forward','backward']),'scoring':None,'cv':random.randint(a=3,b=15),'n_jobs':n_cpu_cores};
    else:
        print(f'Выбрано use_feature_selector==False, feature_selector не применяется');
        feature_selector_hyperparams=None;

    #3.5. Установка типа и гиперпараметров model
    models_having_coef_and_intercept_attributes:list[str]=['LinearRegression','Ridge','SGDRegressor','Lars','Lasso','LassoLars','LassoLarsIC','ARDRegression','BayesianRidge','MultiTaskElasticNet','MultiTaskLasso','HuberRegressor','QuantileRegressor','TheilSenRegressor','GammaRegressor','PoissonRegressor','TweedieRegressor','PassiveAggressiveRegressor','BaggingRegressor','HistGradientBoostingRegressor'];
    if problem_type=='classification':
        if model_type is None:
            #Выбор случайной модели из списка:
            model_types:list[str]=['AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier','HistGradientBoostingClassifier','RandomForestClassifier','XGBClassifier','LGBMClassifier','LogisticRegression','PassiveAggressiveClassifier','Perceptron','RidgeClassifier','SGDClassifier'];
            if prefered_model_types is not None:model_types=list(set(model_types).intersection(set(prefered_model_types)));
            model_type:str=random.choice(seq=model_types);
        if model_hyperparams is None:
            if model_type=='AdaBoostClassifier':model_hyperparams={'n_estimators':random.randint(a=20,b=500),'learning_rate':10**random.uniform(a=-2,b=0.5),'random_state':hyperparam_random_state};
            elif model_type=='BaggingClassifier':model_hyperparams={'n_estimators':random.randint(a=20,b=500),'max_samples':random.uniform(a=0.5,b=0.9),'max_features':random.uniform(a=0.5,b=0.9),'bootstrap':bool(random.randint(a=0,b=1)),'bootstrap_features':bool(random.randint(a=0,b=1)),'oob_score':bool(random.randint(a=0,b=1)),'warm_start':bool(random.randint(a=0,b=1)),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='ExtraTreesClassifier':model_hyperparams={'n_estimators':random.randint(a=20,b=500),'criterion':random.choice(seq=['gini','entropy','log_loss']),'max_depth':random.randint(a=5,b=50),'min_samples_split':random.randint(a=1,b=10),'min_samples_leaf':random.randint(a=1,b=10),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_features':random.choice(seq=['sqrt','log2',None]),'max_leaf_nodes':random.randint(a=5,b=50),'min_impurity_decrease':random.uniform(a=0.0,b=0.1),'bootstrap':bool(random.randint(a=0,b=1)),'oob_score':bool(random.randint(a=0,b=1)),'warm_start':bool(random.randint(a=0,b=1)),'class_weight':random.choice(seq=['balanced','balanced_subsample',None]),'ccp_alpha':random.uniform(a=0.0,b=0.1),'max_samples':random.uniform(a=0.001,b=1.0),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='GradientBoostingClassifier':model_hyperparams={'loss':random.choice(seq=['log_loss','exponential']),'learning_rate':10**random.uniform(a=-5,b=2),'n_estimators':random.randint(a=20,b=500),'subsample':random.uniform(a=0.001,b=1.0),'criterion':random.choice(seq=['friedman_mse','squared_error']),'min_samples_split':random.randint(a=2,b=10),'min_samples_leaf':random.randint(a=1,b=50),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_depth':random.randint(a=1,b=20),'min_impurity_decrease':random.uniform(a=0.0,b=1.0),'init':random.choice(seq=['zero',None]),'max_features':random.choice(seq=['sqrt','log2']),'max_leaf_nodes':random.randint(a=2,b=50),'warm_start':bool(random.randint(a=0,b=1)),'validation_fraction':random.uniform(a=0.00001,b=0.999999),'n_iter_no_change':random.randint(a=1,b=100),'tol':10**random.uniform(a=-8,b=-1),'ccp_alpha':random.uniform(a=0.0,b=0.1),'random_state':hyperparam_random_state};
            elif model_type=='HistGradientBoostingClassifier':model_hyperparams={'learning_rate':10**random.uniform(a=-4,b=0),'max_iter':random.randint(a=20,b=400),'max_leaf_nodes':random.randint(a=2,b=50),'max_depth':random.randint(a=2,b=40),'min_samples_leaf':random.randint(a=5,b=100),'l2_regularization':10**random.uniform(a=-10,b=0),'max_features':random.uniform(a=0.8,b=1.0),'max_bins':random.randint(a=10,b=255),'warm_start':random.choice(seq=[True,False]),'n_iter_no_change':random.randint(a=3,b=30),'tol':10**random.uniform(a=-10,b=-3),'random_state':hyperparam_random_state};
            elif model_type=='RandomForestClassifier':
                model_hyperparams={'n_estimators':random.randint(a=10,b=500),'criterion':random.choice(seq=['gini','entropy','log_loss']),'max_depth':random.randint(a=2,b=50),'min_samples_split':random.randint(a=2,b=50),'min_samples_leaf':random.randint(a=1,b=5),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.2),'max_features':random.choice(seq=['sqrt','log2',None]),'max_leaf_nodes':random.randint(a=10,b=100),'min_impurity_decrease':random.uniform(a=0.0,b=0.1),'bootstrap':random.choice(seq=[True,False]),'oob_score':random.choice(seq=[True,False]),'warm_start':random.choice(seq=[True,False]),'ccp_alpha':random.uniform(a=0.0,b=0.2),'max_samples':random.uniform(a=0.00000001,b=1.0),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
                #if model_hyperparams['bootstrap']==False:model_hyperparams['max_samples']=None;
            elif model_type=='XGBClassifier':model_hyperparams={'n_estimators':random.randint(a=10,b=500),'max_depth':random.randint(a=2,b=40),'max_leaves':random.randint(a=0,b=50),'max_bin':random.randint(a=5,b=100),'grow_policy':random.choice(seq=['depthwise','lossguide']),'learning_rate':10**random.uniform(a=-9,b=-1),'booster':random.choice(seq=['gbtree','gblinear','dart']),'gamma':random.uniform(a=0.0,b=1.0),'min_child_weight':random.uniform(a=0.01,b=0.1),'max_delta_step':random.uniform(a=0.1,b=2.0),'subsample':random.uniform(a=0.01,b=0.99),'sampling_method':random.choice(seq=['uniform','gradient_based']),'colsample_bytree':random.uniform(a=0.5,b=0.99),'colsample_bylevel':random.uniform(a=0.5,b=0.99),'colsample_bynode':random.uniform(a=0.5,b=0.99),'reg_alpha':10**random.uniform(a=-12,b=0),'reg_lambda':10**random.uniform(a=-12,b=0),'num_parallel_tree':random.randint(a=5,b=50),'importance_type':random.choice(seq=['gain','weight','cover','total_gain','total_cover']),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='LGBMClassifier':model_hyperparams={'num_leaves':random.randint(a=10,b=60),'max_depth':random.randint(4,40),'learning_rate':10**random.uniform(a=-4,b=1.5),'n_estimators':random.randint(a=20,b=500),'subsample_for_bin':random.randint(a=50_000,b=500_000),'min_child_weight':random.uniform(a=0.0001,b=0.01),'min_child_samples':random.randint(a=5,b=50),'reg_alpha':10**random.uniform(a=-12,b=0),'reg_lambda':10**random.uniform(a=-12,b=0),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='LogisticRegression':model_hyperparams={'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'dual':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-8,b=0),'C':random.uniform(a=0.5,b=1.5),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']),'max_iter':random.randint(a=50,b=500),'warm_start':random.choice(seq=[True,False]),'l1_ratio':random.uniform(a=0.0,b=1.0),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='PassiveAggressiveClassifier':model_hyperparams={'C':random.uniform(a=0.5,b=1.5),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=3000),'tol':10**random.uniform(a=-7,b=1),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'shuffle':random.choice(seq=[True,False]),'loss':random.choice(seq=['hinge','squared_hinge']),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[True,False,1,2,3,4,5,6,7,8,9,10]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='Perceptron':model_hyperparams={'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-12,b=0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-6,b=0),'shuffle':random.choice(seq=[True,False]),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'warm_start':random.choice(seq=[True,False]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='RidgeClassifier':model_hyperparams={'alpha':10**random.uniform(a=-2,b=1),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-8,b=0),'solver':random.choice(seq=['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']),'positive':random.choice(seq=[True,False]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='SGDClassifier':model_hyperparams={'loss':random.choice(seq=['hinge','log_loss','modified_huber','squared_hinge','perceptron','squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']),'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-8,b=0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-6,b=0),'shuffle':random.choice(seq=[True,False]),'epsilon':10**random.uniform(a=-9,b=3),'learning_rate':random.choice(seq=['constant','optimal','invscaling','adaptive']),'eta0':10**random.uniform(a=-5,b=1),'power_t':random.uniform(a=-5,b=6),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[True,False,1,2,3,4,5,6,7,8,9,10]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
    elif problem_type=='regression':
        if model_type is None:#Если тип модели не указан, то он выбирается случайно из списка model_types
            if task_output=='mono_output':
                model_types:list[str]=['LinearRegression','Ridge','SGDRegressor','ElasticNet','Lars','Lasso','LassoLars','LassoLarsIC','ARDRegression','BayesianRidge','HuberRegressor','QuantileRegressor','RANSACRegressor','TheilSenRegressor','GammaRegressor','PoissonRegressor','TweedieRegressor','PassiveAggressiveRegressor','AdaBoostRegressor','BaggingRegressor','ExtraTreesRegressor','GradientBoostingRegressor','HistGradientBoostingRegressor','RandomForestRegressor'];
            elif task_output=='multi_output':
                model_types:list[str]=['LinearRegression','Ridge','SGDRegressor','ElasticNet','Lars','Lasso','LassoLars','LassoLarsIC','ARDRegression','BayesianRidge','MultiTaskElasticNet','MultiTaskLasso','HuberRegressor','QuantileRegressor','RANSACRegressor','TheilSenRegressor','GammaRegressor','PoissonRegressor','TweedieRegressor','PassiveAggressiveRegressor','AdaBoostRegressor','BaggingRegressor','ExtraTreesRegressor','GradientBoostingRegressor','HistGradientBoostingRegressor','RandomForestRegressor'];
            else:
                print(f'Необходимо задать тип выхода (параметр task_output:str, значения: mono_output или multi_output)');
                return error_str;
            if use_only_linear_models==True:
                model_types:list[str]=list(set(model_types).intersection(set(models_having_coef_and_intercept_attributes)));
                print(f'Выбран параметр use_only_linear_models=True');
            if(non_negative_y_guarantee==False)and('PoissonRegressor'in model_types):model_types.remove('PoissonRegressor');#Some value(s) of y are negative which is not allowed for Poisson regression.
            if prefered_model_types is not None:model_types=list(set(model_types).intersection(set(prefered_model_types)));
            model_type:str=random.choice(seq=model_types);
        print(f'model_types: {model_types}');
        if model_hyperparams is None:#Если словарь гиперпараметров модели не указан, то значение каждого гиперпараметра выбирается случайным образом
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
            elif model_type=='HuberRegressor':model_hyperparams={'epsilon':random.uniform(a=1.0,b=10.0),'max_iter':random.randint(a=20,b=200),'alpha':10**random.uniform(a=-8,b=0),'warm_start':random.choice(seq=[True,False]),'fit_intercept':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-8,b=-2)};
            elif model_type=='QuantileRegressor':
                model_hyperparams={'quantile':random.uniform(a=0.0,b=1.0),'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['highs-ds','highs-ipm','highs','interior-point','revised simplex'])};
                #Solver interior-point is not anymore available in SciPy >= 1.11.0
                if model_hyperparams['solver']=='interior-point':model_hyperparams['solver']=random.choice(seq=['highs-ds','highs-ipm','highs','revised simplex']);
            elif model_type=='RANSACRegressor':model_hyperparams={'min_samples':random.uniform(a=0.0,b=1.0),'max_trials':random.randint(a=50,b=150),'max_skips':random.randint(a=500,b=1000),'stop_n_inliers':random.randint(a=500,b=1000),'stop_score':10**random.uniform(a=3,b=10),'stop_probability':random.uniform(a=0.95,b=1.00),'loss':random.choice(seq=['absolute_error','squared_error']),'random_state':hyperparam_random_state};
            elif model_type=='TheilSenRegressor':model_hyperparams={'fit_intercept':random.choice(seq=[True,False]),'max_subpopulation':10**random.uniform(a=-6,b=-2),'n_subsamples':random.randint(a=opened_data.shape[1]+1,b=opened_data.shape[0]),'max_iter':random.randint(a=100,b=500),'tol':10**random.uniform(a=-5,b=-1),'n_jobs':-1,'random_state':hyperparam_random_state};
            elif model_type=='GammaRegressor':model_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-9,b=-2),'warm_start':random.choice(seq=[True,False])};
            elif model_type=='PoissonRegressor':model_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-6,b=-2),'warm_start':random.choice(seq=[True,False])};
            elif model_type=='TweedieRegressor':model_hyperparams={'power':random.choice(seq=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.01,1.02,1.03,1.04,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,1.96,1.97,1.98,1.99,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0]),'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'link':random.choice(seq=['auto','identity','log']),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-6,b=-2),'warm_start':random.choice(seq=[True,False])};
            elif model_type=='PassiveAggressiveRegressor':model_hyperparams={'C':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=2000),'tol':10**random.uniform(a=-5,b=-1),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=0.9),'n_iter_no_change':random.randint(a=2,b=10),'loss':random.choice(seq=['epsilon_insensitive','squared_epsilon_insensitive']),'epsilon':random.uniform(a=0.05,b=0.15),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[False,False,False,False,False,False,False,False,False,False,False,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]),'random_state':hyperparam_random_state};
            elif model_type=='AdaBoostRegressor':model_hyperparams={'n_estimators':random.randint(a=10,b=500),'learning_rate':10**random.uniform(a=-3,b=2),'loss':random.choice(seq=['linear','square','exponential']),'random_state':hyperparam_random_state};
            elif model_type=='BaggingRegressor':
                model_hyperparams={'n_estimators':random.randint(a=5,b=30),'max_samples':random.uniform(a=0.2,b=1.0),'max_features':random.uniform(a=0.2,b=1.0),'bootstrap':random.choice(seq=[True,False]),'bootstrap_features':random.choice(seq=[True,False]),'oob_score':random.choice(seq=[True,False]),'warm_start':random.choice(seq=[True,False]),'n_jobs':n_cpu_cores,'random_state':hyperparam_random_state};
                #Out of bag estimation only available if bootstrap=True
                if model_hyperparams['bootstrap']==False:model_hyperparams['oob_score']=False;
                #Out of bag estimate only available if warm_start=False
                if model_hyperparams['warm_start']==True:model_hyperparams['oob_score']=False;

            elif model_type=='ExtraTreesRegressor':model_hyperparams={'n_estimators':random.randint(a=20,b=200),'criterion':random.choice(seq=['squared_error','absolute_error','friedman_mse','poisson']),'max_depth':random.randint(a=5,b=20),'min_samples_split':random.uniform(a=0.0,b=1.0),'min_samples_leaf':random.uniform(a=0.0,b=1.0),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_features':random.choice(seq=['sqrt','log2',None]),'max_leaf_nodes':random.randint(a=5,b=10),'min_impurity_decrease':random.uniform(a=0.0,b=0.5),'bootstrap':random.choice(seq=[True,False]),'oob_score':random.choice(seq=[True,False]),'n_jobs':n_cpu_cores,'warm_start':random.choice(seq=[True,False]),'ccp_alpha':random.uniform(a=0.0,b=0.1),'max_samples':random.uniform(a=0.0,b=1.0),'random_state':hyperparam_random_state};
            elif model_type=='GradientBoostingRegressor':model_hyperparams={'loss':random.choice(seq=['squared_error','absolute_error','huber','quantile']),'learning_rate':10**random.uniform(a=-2,b=0),'n_estimators':random.randint(a=20,b=300),'subsample':random.uniform(a=0.0,b=1.0),'criterion':random.choice(seq=['friedman_mse','squared_error']),'min_samples_split':random.uniform(a=0.0,b=1.0),'min_samples_leaf':random.uniform(a=0.0,b=1.0),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_depth':random.randint(a=1,b=7),'min_impurity_decrease':random.uniform(a=0.0,b=1.0),'max_features':random.choice(seq=['sqrt','log2','sqrt','log2','sqrt','log2','sqrt','log2',0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,None,None,None,None,None,None]),'alpha':random.uniform(a=0.0,b=1.0),'max_leaf_nodes':random.randint(a=2,b=100),'warm_start':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=0.4),'n_iter_no_change':random.randint(a=5,b=20),'tol':10**random.uniform(a=-6,b=-2),'ccp_alpha':random.uniform(a=0.0,b=100.0),'random_state':hyperparam_random_state};
            elif model_type=='HistGradientBoostingRegressor':
                model_hyperparams={'loss':random.choice(seq=['squared_error','absolute_error','gamma','poisson','quantile']),'quantile':random.uniform(a=0.0,b=1.0),'learning_rate':10**random.uniform(a=-2,b=0),'max_iter':random.randint(a=20,b=200),'max_leaf_nodes':random.randint(a=2,b=60),'max_depth':random.randint(a=2,b=10),'min_samples_leaf':random.randint(a=5,b=50),'l2_regularization':random.uniform(a=0.0,b=1.0),'max_features':random.uniform(a=0.2,b=1.0),'max_bins':random.randint(a=10,b=255),'warm_start':random.choice(seq=[True,False]),'early_stopping':random.choice(seq=['auto',True]),'scoring':random.choice(seq=['loss',None]),'validation_fraction':random.uniform(a=0.05,b=0.25),'n_iter_no_change':random.randint(a=3,b=30),'tol':10**random.uniform(a=-11,b=-3),'random_state':hyperparam_random_state};
                #loss='poisson' requires non-negative y and sum(y) > 0
                if non_negative_y_guarantee==False:model_hyperparams['loss']=random.choice(seq=['squared_error','absolute_error','gamma','quantile']);
            elif model_type=='RandomForestRegressor':
                model_hyperparams={'n_estimators':random.randint(a=20,b=200),'criterion':random.choice(seq=['squared_error','absolute_error','friedman_mse','poisson']),'max_depth':random.randint(a=2,b=20),'min_samples_split':random.uniform(a=0.0,b=1.0),'min_samples_leaf':random.uniform(a=0.0,b=1.0),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_features':random.choice(seq=['sqrt','log2','sqrt','log2','sqrt','log2','sqrt','log2','sqrt','log2',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),'min_impurity_decrease':random.uniform(a=0.0,b=0.2),'bootstrap':random.choice(seq=[True,False]),'warm_start':random.choice(seq=[True,False]),'ccp_alpha':random.uniform(a=0.0,b=1.0),'max_samples':random.uniform(a=0.0,b=1.0),'n_jobs':n_cpu_cores,'random_state':hyperparam_random_state};
                #`max_sample` cannot be set if `bootstrap=False`. Either switch to `bootstrap=True` or set `max_sample=None`
                if model_hyperparams['bootstrap']==False:model_hyperparams['max_sample']=None;
                #Some value(s) of y are negative which is not allowed for Poisson regression
                if non_negative_y_guarantee==False:model_hyperparams['criterion']=random.choice(seq=['squared_error','absolute_error','friedman_mse']);
    
    #3.6. Инициализация словарей для хранения конструкторов для scaler, feature_selector, model
    imputer_constructors_dict:dict[str:AnyImputer]={'KNNImputer':KNNImputer,'SimpleImputer':SimpleImputer};
    scaler_constructors_dict:dict[str:AnyScaler]={'MaxAbsScaler':MaxAbsScaler,'MinMaxScaler':MinMaxScaler,'RobustScaler':RobustScaler,'StandardScaler':StandardScaler};
    fs_estimator_constructors_dict:dict[str:AnyFSEstimator]={'LogisticRegression':LogisticRegression,'PassiveAggressiveClassifier':PassiveAggressiveClassifier,'Perceptron':Perceptron,'RidgeClassifier':RidgeClassifier,'SGDClassifier':SGDClassifier,'LinearRegression':LinearRegression,'Ridge':Ridge,'SGDRegressor':SGDRegressor,'ElasticNet':ElasticNet,'Lars':Lars,'Lasso':Lasso,'LassoLars':LassoLars,'LassoLarsIC':LassoLarsIC,'OrthogonalMatchingPursuit':OrthogonalMatchingPursuit,'ARDRegression':ARDRegression,'BayesianRidge':BayesianRidge,'HuberRegressor':HuberRegressor,'QuantileRegressor':QuantileRegressor,'RANSACRegressor':RANSACRegressor,'TheilSenRegressor':TheilSenRegressor,'GammaRegressor':GammaRegressor,'PoissonRegressor':PoissonRegressor,'TweedieRegressor':TweedieRegressor,'PassiveAggressiveRegressor':PassiveAggressiveRegressor,'LinearSVC':LinearSVC,'NuSVC':NuSVC,'SVC':SVC,'LinearSVR':LinearSVR,'NuSVR':NuSVR,'SVR':SVR};
    feature_selector_constructors_dict:dict[str:AnyFeatureSelector]={'GenericUnivariateSelect':GenericUnivariateSelect,'RFE':RFE,'RFECV':RFECV,'SelectFdr':SelectFdr,'SelectFpr':SelectFpr,'SelectFromModel':SelectFromModel,'SelectFwe':SelectFwe,'SelectKBest':SelectKBest,'SelectPercentile':SelectPercentile,'SequentialFeatureSelector':SequentialFeatureSelector};
    model_constructors_dict:dict[str:AnyModel]={'LogisticRegression':LogisticRegression,'PassiveAggressiveClassifier':PassiveAggressiveClassifier,'Perceptron':Perceptron,'RidgeClassifier':RidgeClassifier,'SGDClassifier':SGDClassifier,'AdaBoostClassifier':AdaBoostClassifier,'BaggingClassifier':BaggingClassifier,'ExtraTreesClassifier':ExtraTreesClassifier,'GradientBoostingClassifier':GradientBoostingClassifier,'HistGradientBoostingClassifier':HistGradientBoostingClassifier,'RandomForestClassifier':RandomForestClassifier,'XGBClassifier':XGBClassifier,'LGBMClassifier':LGBMClassifier,'LinearRegression':LinearRegression,'Ridge':Ridge,'SGDRegressor':SGDRegressor,'ElasticNet':ElasticNet,'Lasso':Lasso,'LassoLarsIC':LassoLarsIC,'ARDRegression':ARDRegression,'OrthogonalMatchingPursuit':OrthogonalMatchingPursuit,'BayesianRidge':BayesianRidge,'MultiTaskElasticNet':MultiTaskElasticNet,'MultiTaskLasso':MultiTaskLasso,'HuberRegressor':HuberRegressor,'QuantileRegressor':QuantileRegressor,'RANSACRegressor':RANSACRegressor,'Lars':Lars,'TheilSenRegressor':TheilSenRegressor,'GammaRegressor':GammaRegressor,'PoissonRegressor':PoissonRegressor,'TweedieRegressor':TweedieRegressor,'PassiveAggressiveRegressor':PassiveAggressiveRegressor,'LassoLars':LassoLars,'AdaBoostRegressor':AdaBoostRegressor,'BaggingRegressor':BaggingRegressor,'ExtraTreesRegressor':ExtraTreesRegressor,'GradientBoostingRegressor':GradientBoostingRegressor,'HistGradientBoostingRegressor':HistGradientBoostingRegressor,'RandomForestRegressor':RandomForestRegressor};
    


    # 4. Подготовка кросс-валидации
    if problem_type=='classification':K_Fold:StratifiedKFold=StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=split_random_state)
    elif problem_type=='regression':K_Fold:KFold=KFold(n_splits=num_folds,shuffle=True,random_state=split_random_state);
    #StratifiedKFold предназначен ТОЛЬКО для классификационных задач, где целевая переменная имеет дискретные значения (бинарные или 
    #мультикласс). В регрессии же целевая переменная непрерывная (continuous), и StratifiedKFold не может работать с такими данными.
    
    scores_valid:list[float]=[];
    print(f"""Начинаем обработку пайплайна со следующими значениями: n_features_all: {n_features_all}, n_features_selected_randomly: {n_features_selected_randomly}
    randomly_selected_indexes: {randomly_selected_indexes}
    use_var_thresholder: {use_var_thresholder}, var_thresholder_type: {var_thresholder_type}, var_thresholder_hyperparams: {var_thresholder_hyperparams}
    use_scaler: {use_scaler}, scaler_type: {scaler_type}, scaler_hyperparams: {scaler_hyperparams}
    use_feature_selector: {use_feature_selector}, feature_selector_type: {feature_selector_type}
    feature_selector_hyperparams: {feature_selector_hyperparams}
    fs_score_func_type: {fs_score_func_type}, fs_estimator_type: {fs_estimator_type}
    fs_estimator_hyperparams: {fs_estimator_hyperparams}
    model_type: {model_type}
    model_hyperparams: {model_hyperparams}
    split_random_state: {split_random_state}
    score_type: {score_type}""");
    for fold_num,(train_index,valid_index) in enumerate(K_Fold.split(X=X_train_cv,y=y_train_cv)):
        print(f"  Обрабатываем фолд {fold_num+1}/{num_folds}...",end=' ');

        # 4.0 Разделение на train/valid для фолда
        X_train_fold:np.ndarray=X_train_cv[train_index];y_train_fold:np.ndarray=y_train_cv[train_index];
        X_valid_fold:np.ndarray=X_train_cv[valid_index];y_valid_fold:np.ndarray=y_train_cv[valid_index];

        print(f'Доля X_train_fold от opened_data: {X_train_fold.shape[0]/opened_data_len}',end='; ');#0.72
        print(f'Доля X_valid_fold от opened_data: {X_valid_fold.shape[0]/opened_data_len}',end='; ');#0.08

        #4.1. Использование imputer (если выбрано) для cross_valid
        if use_imputer==True:
            imputer_cross_valid:AnyImputer=imputer_constructors_dict[imputer_type](**imputer_hyperparams);
            imputer_cross_valid.fit(X=X_train_fold,y=y_train_fold);
            X_train_fold_imputed=imputer_cross_valid.transform(X=X_train_fold);
            X_valid_fold_imputed=imputer_cross_valid.transform(X=X_valid_fold);
        else:
            X_train_fold_imputed=X_train_fold.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
            X_valid_fold_imputed=X_valid_fold.copy();
            imputer_cross_valid=None;

        #4.2. Использование var_thresholder (если выбрано) для cross_valid
        if use_var_thresholder==True:
            if var_thresholder_type=='VarianceThreshold':var_thresholder_cross_valid=VarianceThreshold(**var_thresholder_hyperparams);
            var_thresholder_cross_valid.fit(X=X_train_fold_imputed,y=y_train_fold);
            X_train_fold_var_thresholded=var_thresholder_cross_valid.transform(X=X_train_fold_imputed);
            X_valid_fold_var_thresholded=var_thresholder_cross_valid.transform(X=X_valid_fold_imputed);
        else:
            X_train_fold_var_thresholded=X_train_fold_imputed.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
            X_valid_fold_var_thresholded=X_valid_fold_imputed.copy();
            var_thresholder_cross_valid=None;

        #4.3. Использование scaler (если выбрано) для cross_valid
        if use_scaler==True:
            scaler_cross_valid:AnyScaler=scaler_constructors_dict[scaler_type](**scaler_hyperparams);
            scaler_cross_valid.fit(X=X_train_fold_var_thresholded);#fit только на train, без valid
            X_train_fold_scaled=scaler_cross_valid.transform(X=X_train_fold_var_thresholded);
            X_valid_fold_scaled=scaler_cross_valid.transform(X=X_valid_fold_var_thresholded);
        else:#Если use_scaler==False
            X_train_fold_scaled=X_train_fold_var_thresholded.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
            X_valid_fold_scaled=X_valid_fold_var_thresholded.copy();
            scaler_cross_valid=None;

        #4.4. Использование feature_selector (если выбрано) для cross_valid
        if use_feature_selector==True:
            if feature_selector_type in ['GenericUnivariateSelect','SelectFdr','SelectFpr','SelectFwe','SelectKBest','SelectPercentile']:
                if fs_score_func_type=='f_classif':score_func_cross_valid:callable=f_classif;
                elif fs_score_func_type=='mutual_info_classif':score_func_cross_valid:callable=mutual_info_classif;
                elif fs_score_func_type=='chi2':score_func_cross_valid:callable=chi2;
                elif fs_score_func_type=='f_regression':score_func_cross_valid:callable=f_regression;
                elif fs_score_func_type=='mutual_info_regression':score_func_cross_valid:callable=mutual_info_regression;
                feature_selector_cross_valid:AnyFeatureSelector=feature_selector_constructors_dict[feature_selector_type](**feature_selector_hyperparams);
            elif feature_selector_type in feature_selector_types_estimator_all:
                fs_estimator_cross_valid:AnyFSEstimator=fs_estimator_constructors_dict[fs_estimator_type](**fs_estimator_hyperparams);
                feature_selector_cross_valid:AnyFeatureSelector=feature_selector_constructors_dict[feature_selector_type](estimator=fs_estimator_cross_valid,**feature_selector_hyperparams);
            feature_selector_cross_valid.fit(X=X_train_fold_scaled,y=y_train_fold);
            X_train_fold_feature_selected=feature_selector_cross_valid.transform(X=X_train_fold_scaled);
            X_valid_fold_feature_selected=feature_selector_cross_valid.transform(X=X_valid_fold_scaled);
        else:#Если use_feature_selector==False
            X_train_fold_feature_selected=X_train_fold_scaled.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
            X_valid_fold_feature_selected=X_valid_fold_scaled.copy();
            feature_selector_cross_valid=None;
        if (X_train_fold_feature_selected.shape[1]==0)or(X_valid_fold_feature_selected.shape[1]==0):
            print(f'После применения feature_selector типа {feature_selector_type} на этапе cross_valid количество отобранных признаков равно нулю, обработка этого пайплайна прервана');
            return error_str;

        #4.5. Использование model (всегда выбрано) для cross_valid
        model_cross_valid:AnyModel=model_constructors_dict[model_type](**model_hyperparams);
        model_cross_valid.fit(X=X_train_fold_feature_selected,y=y_train_fold);
        y_valid_pred=model_cross_valid.predict(X_valid_fold_feature_selected);
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
                print(f'Значение метрики ниже установленного минимального порога ({score_valid_min_threshold:.6f}), валидация этого пайплайна с этим набором гиперпараметров прервана для экономии времени');
                return error_str;
        elif score_valid_max_threshold!=None:
            if score_valid>score_valid_max_threshold:
                print(f'Значение метрики выше установленного максимального порога ({score_valid_max_threshold:.6f}), валидация этого пайплайна с этим набором гиперпараметров прервана для экономии времени');
                return error_str;
        scores_valid.append(score_valid);
        

    # 5. Расчет среднего значения метрики по CV (mean) и её среднеквадратического отклонения (std) [mean чем больше тем лучше или чем
    #меньше тем лучше - в зависимости от метрики, std всегда чем меньше тем лучше, так как чем меньше std, тем устойчивее пайплайн]
    score_valid_mean:float=np.mean(a=scores_valid,dtype=np.float64);
    score_valid_std:float=np.std(a=scores_valid,dtype=np.float64,ddof=0);
    print(f"\nСреднее значение метрики {score_type} по кросс-валидации: {score_valid_mean:.6f}, среднеквадратическое отклонение метрики {score_type} по кросс-валидации: {score_valid_std:.6f}");

    # 6. Оценка качества на отложенной выборке (20% открытых данных)
    #print("Оценка качества на отложенной выборке (20% открытых данных)...")

    #6.1. Использование imputer (если выбрано) для for_final_test
    if use_imputer==True:
        imputer_for_final_test:AnyImputer=imputer_constructors_dict[imputer_type](**imputer_hyperparams);
        imputer_for_final_test.fit(X=X_train_cv,y=y_train_cv);
        X_train_cv_imputed=imputer_for_final_test.transform(X=X_train_cv);
        X_test_final_imputed=imputer_for_final_test.transform(X=X_test_final);
    else:
        X_train_cv_imputed=X_train_cv.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
        X_test_final_imputed=X_test_final.copy();
        imputer_for_final_test=None;

    #6.2. Использование var_thresholder (если выбрано) для for_final_test
    if use_var_thresholder==True:
        if var_thresholder_type=='VarianceThreshold':var_thresholder_for_final_test=VarianceThreshold(**var_thresholder_hyperparams);
        var_thresholder_for_final_test.fit(X=X_train_cv_imputed,y=y_train_cv);
        X_train_cv_var_thresholded=var_thresholder_for_final_test.transform(X=X_train_cv_imputed);
        X_test_final_var_thresholded=var_thresholder_for_final_test.transform(X=X_test_final_imputed);
    else:
        X_train_cv_var_thresholded=X_train_cv_imputed.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
        X_test_final_var_thresholded=X_test_final_imputed.copy();
        var_thresholder_for_final_test=None;

    #6.3. Использование scaler (если выбрано) для for_final_test
    if use_scaler==True:
        scaler_for_final_test:AnyScaler=scaler_constructors_dict[scaler_type](**scaler_hyperparams);
        scaler_for_final_test.fit(X=X_train_cv_var_thresholded);#fit только на train, без valid
        X_train_cv_scaled=scaler_for_final_test.transform(X=X_train_cv_var_thresholded);
        X_test_final_scaled=scaler_for_final_test.transform(X=X_test_final_var_thresholded);
    else:#Если use_scaler==False
        X_train_cv_scaled=X_train_cv_var_thresholded.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
        X_test_final_scaled=X_test_final_var_thresholded.copy();
        scaler_for_final_test=None;


    #6.4. Использование feature_selector (если выбрано) для for_final_test
    if use_feature_selector==True:
        if feature_selector_type in ['GenericUnivariateSelect','SelectFdr','SelectFpr','SelectFwe','SelectKBest','SelectPercentile']:
            if fs_score_func_type=='f_classif':score_func_for_final_test:callable=f_classif;
            elif fs_score_func_type=='mutual_info_classif':score_func_for_final_test:callable=mutual_info_classif;
            elif fs_score_func_type=='chi2':score_func_for_final_test:callable=chi2;
            elif fs_score_func_type=='f_regression':score_func_for_final_test:callable=f_regression;
            elif fs_score_func_type=='mutual_info_regression':score_func_for_final_test:callable=mutual_info_regression;
            feature_selector_for_final_test:AnyFeatureSelector=feature_selector_constructors_dict[feature_selector_type](**feature_selector_hyperparams);
        elif feature_selector_type in feature_selector_types_estimator_all:
            fs_estimator_for_final_test:AnyFSEstimator=fs_estimator_constructors_dict[fs_estimator_type](**fs_estimator_hyperparams);
            feature_selector_for_final_test:AnyFeatureSelector=feature_selector_constructors_dict[feature_selector_type](estimator=fs_estimator_for_final_test,**feature_selector_hyperparams);
        feature_selector_for_final_test.fit(X=X_train_cv_scaled,y=y_train_cv);
        X_train_cv_feature_selected=feature_selector_for_final_test.transform(X=X_train_cv_scaled);
        X_test_final_feature_selected=feature_selector_for_final_test.transform(X=X_test_final_scaled);
    else:#Если use_feature_selector==False
        X_train_cv_feature_selected=X_train_cv_scaled.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
        X_test_final_feature_selected=X_test_final_scaled.copy();
        feature_selector_for_final_test=None;
    if (X_train_cv_feature_selected.shape[1]==0)or(X_test_final_feature_selected.shape[1]==0):
        print(f'После применения feature_selector типа {feature_selector_type} на этапе final_test количество отобранных признаков равно нулю, обработка этого пайплайна прервана');
        return error_str;
    
    #6.5. Использование model (всегда выбрано) для for_final_test
    model_for_final_test:AnyModel=model_constructors_dict[model_type](**model_hyperparams);
    model_for_final_test.fit(X=X_train_cv_feature_selected,y=y_train_cv);#Обучение на 80% открытых данных
    y_test_pred=model_for_final_test.predict(X_test_final_feature_selected);#Тестирование на 20% открытых данных
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
    
    #7. ОБУЧЕНИЕ ФИНАЛЬНОГО ПАЙПЛАЙНА НА 100% ОТКРЫТЫХ ДАННЫХ И СОХРАНЕНИЕ ПАЙПЛАЙНА В *.pkl файл
    print("Обучение финального production-пайплайна на ВСЕХ данных...");
    X_all:np.ndarray=opened_data.copy();y_all:np.ndarray=opened_target.copy();

    #7.1. Использование imputer (если выбрано) для production
    if use_imputer==True:
        imputer_production:AnyImputer=imputer_constructors_dict[imputer_type](**imputer_hyperparams);
        imputer_production.fit(X=X_all,y=y_all);
        X_all_imputed=imputer_production.transform(X=X_all);
    else:
        X_all_imputed=X_all.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
        imputer_production=None;

    #7.2. Использование var_thresholder (если выбрано) для production
    if use_var_thresholder==True:
        if var_thresholder_type=='VarianceThreshold':var_thresholder_production=VarianceThreshold(**var_thresholder_hyperparams);
        var_thresholder_production.fit(X=X_all_imputed,y=y_all);
        X_all_var_thresholded=var_thresholder_production.transform(X=X_all_imputed);
    else:
        X_all_var_thresholded=X_all_imputed.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
        var_thresholder_production=None;

    #7.3. Использование scaler (если выбрано) для production
    if use_scaler==True:
        scaler_production:AnyScaler=scaler_constructors_dict[scaler_type](**scaler_hyperparams);
        scaler_production.fit(X=X_all_var_thresholded);
        X_all_scaled=scaler_production.transform(X=X_all_var_thresholded);
    else:
        X_all_scaled=X_all_var_thresholded.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
        scaler_production=None;#Переменной scaler_production нужно что-то присвоить чтобы сохранить в pkl файле

    #7.4. Использование feature_selector (если выбрано) для production
    if use_feature_selector==True:
        if feature_selector_type in ['GenericUnivariateSelect','SelectFdr','SelectFpr','SelectFwe','SelectKBest','SelectPercentile']:
            if fs_score_func_type=='f_classif':score_func_production:callable=f_classif;
            elif fs_score_func_type=='mutual_info_classif':score_func_production:callable=mutual_info_classif;
            elif fs_score_func_type=='chi2':score_func_production:callable=chi2;
            elif fs_score_func_type=='f_regression':score_func_production:callable=f_regression;
            elif fs_score_func_type=='mutual_info_regression':score_func_production:callable=mutual_info_regression;
            feature_selector_production:AnyFeatureSelector=feature_selector_constructors_dict[feature_selector_type](**feature_selector_hyperparams);
        elif feature_selector_type in feature_selector_types_estimator_all:
            fs_estimator_production:AnyFSEstimator=fs_estimator_constructors_dict[fs_estimator_type](**fs_estimator_hyperparams);
            feature_selector_production:AnyFeatureSelector=feature_selector_constructors_dict[feature_selector_type](estimator=fs_estimator_production,**feature_selector_hyperparams);
        feature_selector_production.fit(X=X_all_scaled,y=y_all);
        X_all_feature_selected=feature_selector_production.transform(X=X_all_scaled);
    else:#Если use_feature_selector==False
        X_all_feature_selected=X_all_scaled.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
        feature_selector_production=None;#Переменной feature_selector_production нужно что-то присвоить чтобы сохранить в pkl файле
    if X_all_feature_selected.shape[1]==0:
        print(f'После применения feature_selector типа {feature_selector_type} на этапе production количество отобранных признаков равно нулю, обработка этого пайплайна прервана');
        return error_str;

    #7.5. Использование model (всегда выбрано) для production
    model_production:AnyModel=model_constructors_dict[model_type](**model_hyperparams);
    model_production.fit(X=X_all_feature_selected,y=y_all);

    # 8. Генерация ID пайплайна и сохранение Production-пайплайна
    pipeline_id:str=''.join(random.choices(population=string.ascii_uppercase+string.digits,k=16));
    pipeline_filename:str=f"pipeline_{pipeline_id}.pkl";

    # Сохраняем production-пайплайн (именно production imputer, var_thresholder, scaler, feature_selector, model)
    with open(file=pipeline_filename,mode='wb')as f:pickle.dump(obj={'n_features_selected_randomly':n_features_selected_randomly,'randomly_selected_indexes':randomly_selected_indexes,'imputer':imputer_production,'var_thresholder':var_thresholder_production,'scaler':scaler_production,'feature_selector':feature_selector_production,'fs_score_func_type':fs_score_func_type,'fs_estimator_type':fs_estimator_type,'model':model_production},file=f);
    pipeline_file_size:int=os.path.getsize(filename=pipeline_filename);
    scores_valid_str:str='['+', '.join([f"{s:.6f}" for s in scores_valid])+']';

    seconds_pipe_finish:float=time.time();#Для лога (чтобы вычислить время обработки этого пайплайна)
    seconds_processing:float=seconds_pipe_finish-seconds_pipe_start;

    # 9. Логирование
    log_record_txt:str=f"""
--- pipeline ID: {pipeline_id} ---
n_features_all: {n_features_all}
n_features_selected_randomly: {n_features_selected_randomly}
randomly_selected_indexes: {randomly_selected_indexes}
use_imputer: {use_imputer}
imputer_type: {imputer_type}
imputer_hyperparams: {imputer_hyperparams}
use_var_thresholder: {use_var_thresholder}
var_thresholder_type: {var_thresholder_type}
var_thresholder_hyperparams: {var_thresholder_hyperparams}
use_scaler: {use_scaler}
scaler_type: {scaler_type}
scaler_hyperparams: {scaler_hyperparams}
use_feature_selector: {use_feature_selector}
feature_selector_type: {feature_selector_type}
feature_selector_hyperparams: {feature_selector_hyperparams}
fs_score_func_type: {fs_score_func_type}
fs_estimator_type: {fs_estimator_type}
fs_estimator_hyperparams: {fs_estimator_hyperparams}
model_type: {model_type}
model_hyperparams: {model_hyperparams}
split_random_state: {split_random_state}
score_type: {score_type}
scores_valid: {scores_valid_str}
score_valid_mean: {score_valid_mean}
score_valid_std: {score_valid_std}
score_test: {score_test}
dt_pipe_start_str: {dt_pipe_start_str}
seconds_pipe_start: {seconds_pipe_start}, seconds_pipe_finish: {seconds_pipe_finish}, seconds_processing: {seconds_processing}
pipeline_file_size (bytes): {pipeline_file_size}
---------------------------------------
"""
    #Никакие гиперпараметры (feature_selector_hyperparams, scaler_hyperparams, model_hyperparams) не записываются в csv файл, так как они
    #представляют собой словарь (dict), пары key:value которого записываются через запятую, что нарушило бы консистентность строк csv файла,
    #потому что эти словари у разных пайплайнов имеют разное количество значений (у разных типов моделей разное количество киперпараметров)
    log_record_csv:str=f'{pipeline_id},{n_features_all},{n_features_selected_randomly},{use_imputer},{imputer_type},{use_var_thresholder},{var_thresholder_type},{use_scaler},{scaler_type},{use_feature_selector},{feature_selector_type},{fs_score_func_type},{fs_estimator_type},{model_type},{score_type},{score_valid_mean},{score_valid_std},{score_test},{dt_pipe_start_str},{seconds_processing},{pipeline_file_size}\n';
    print(log_record_txt);
    with open(file='log_pipelines.txt',mode='at',encoding='UTF-8')as log_file:log_file.write(log_record_txt);
    with open(file='log_pipelines.csv',mode='at',encoding='UTF-8')as log_file:log_file.write(log_record_csv);
    return pipeline_id;

def float_list_to_comma_separated_str(float_list:list[float],digits:int=2):
    float_list_buf=list(np.round(np.array(float_list),digits));
    return ','.join([str(x)for x in float_list_buf]);

def create_predictions_files(pipeline_ids:list[str],digits_round_min:int=1,digits_round_max:int=18,print_all_predictions:bool=False)->None:
    '''Функция вычисляет предсказания и сохраняет их в файлах tsv и json, позволяя выбирать количество цифр'''
    targets_dict:dict[str:float]={};
    for sample_id in closed_ids:
        targets_dict[sample_id]=0.0;
    print(f'targets_dict: {targets_dict}');
    print(f'len(targets_dict): {len(targets_dict)}');
    pipeline_ids_str:str=' '.join(pipeline_ids);
    results_file_id:str=''.join(random.choices(population=string.ascii_uppercase+string.digits,k=16));
    n_pipelines:int=len(pipeline_ids);
    print(f'Предсказания выполняются усреднением результатов для n_pipelines={n_pipelines} пайплайнов со значениями id: {pipeline_ids}');
    buf_s:str='';
    n_samples_opened:int=opened_ids.shape[0];
    n_samples_closed:int=closed_ids.shape[0];
    targets_list:list[float]=[0.0 for i in range(n_samples_closed)];#Изначально предсказания для всех образцов инициализируем нулями
    #так как 0+a=a для любого действительного числа a, все предсказания - это числа с плавающей точкой
    #Суммирование предсказаний пайплайнов:
    for pipeline_id in pipeline_ids:#Перебираем пайплайны
        pkl_file_name:str=f'pipeline_{pipeline_id}.pkl';
        with open(file=pkl_file_name,mode='rb')as f:pipeline_dict:dict=pickle.load(file=f);
        print(f'pipeline_id: {pipeline_id}, pipeline_dict: {pipeline_dict}');
        n_features_selected_randomly:int=pipeline_dict['n_features_selected_randomly'];
        randomly_selected_indexes:list[int]=pipeline_dict['randomly_selected_indexes'];
        imputer:AnyImputer=pipeline_dict['imputer'];#imputer можкт быть None
        var_thresholder:VarianceThreshold=pipeline_dict['var_thresholder'];#var_thresholder может быть None
        scaler:AnyScaler|None=pipeline_dict['scaler'];#scaler может быть None
        feature_selector:AnyFeatureSelector|None=pipeline_dict['feature_selector'];#feature_selector может быть None
        model:AnyModel=pipeline_dict['model'];#model не может быть None
        print(f'pipeline_id: {pipeline_id}');
        print(f'model.__dict__: {model.__dict__}');#model точно не None
        if feature_selector is not None:print(f'feature_selector.__dict__: {feature_selector.__dict__}');
        else:print(f'feature_selector: {feature_selector}');        
        if scaler is not None:print(f'scaler.__dict__: {scaler.__dict__}');
        else:print(f'scaler: {scaler}');        
        num_processed:int=0;
        opened_data:np.ndarray=np.zeros((n_samples_opened,n_features_selected_randomly));#Создание новых массивов
        closed_data:np.ndarray=np.zeros((n_samples_closed,n_features_selected_randomly));
        #Использование списка randomly_selected_indexes и его сохранение в лог делает эсперименты воспроизводимыми
        for col_idx,feature_idx in enumerate(randomly_selected_indexes):#Копирование столбцов с выбранными индексами из исходных массивов
            opened_data[:,col_idx]=opened_data_all_features[:,feature_idx];
            closed_data[:,col_idx]=closed_data_all_features[:,feature_idx];
        for sample_num in range(n_samples_closed):#Перебираем образцы закрытых данных
            sample_id:str=closed_ids[sample_num];
            features:np.ndarray=closed_data[sample_num].reshape(1, -1);#
            #1. Использование imputer (если есть в пайплайне) для prediction
            if imputer is not None:features_imputed=imputer.transform(X=features);
            else:features_imputed=features.copy();
            #2. Использование var_thresholder (если есть в пайплайне) для prediction
            if var_thresholder is not None:features_var_thresholded=var_thresholder.transform(X=features_imputed);
            else:features_var_thresholded=features_imputed.copy();
            #3. Использование scaler (если есть в пайплайне) для prediction
            if scaler is not None:features_scaled=scaler.transform(X=features_var_thresholded);
            else:features_scaled=features_var_thresholded.copy();
            #4. Использование feature_selector (если есть в пайплайне) для prediction
            if feature_selector is not None:features_feature_selected=feature_selector.transform(X=features_scaled);
            else:features_feature_selected=features_scaled.copy();
            #5. Использование model (всегда есть в пайплайне) для prediction
            target_predicted:float=model.predict(features_feature_selected)[0];
            num_processed=num_processed+1;
            #targets_list.append(target_predicted);
            targets_list[sample_num]=targets_list[sample_num]+target_predicted;
            if print_all_predictions==True:
                print(f'sample_num: {sample_num:5d}, sample_id: {sample_id}, target_predicted: {target_predicted}, type(target_predicted): {type(target_predicted)}');
    #Приведение списка targets_list из list[numpy.float64] в list[float]
    for sample_num in range(n_samples_closed):targets_list[sample_num]=float(targets_list[sample_num]);
    #Усреднение предсказаний пайплайнов:
    for sample_num in range(n_samples_closed):
        targets_list[sample_num]=targets_list[sample_num]/n_pipelines;#Деление суммы предсказаний пайплайнов на количество пайплайнов
        buf_s=buf_s+closed_ids[sample_num]+'\t'+str(targets_list[sample_num])+'\n';#Добавление информации в строку для вывода в tsv файл
    tsv_filename:str='result_'+results_file_id+'.tsv';
    with open(file=tsv_filename,mode='wt',encoding='UTF-8')as tsv_file:tsv_file.write(buf_s);    
    #targets_ndarray:np.ndarray=np.ndarray(shape=(n_samples_closed,),dtype=np.float32);
    #for i in range(n_samples_closed):targets_ndarray[i]=targets_list[i];
    #targets_ndarray=targets_ndarray.round(decimals=2);
    for dig in range(digits_round_min,digits_round_max+1):#Чтобы можно было создать несколько json файлов с разным количеством цифр
        predictions_str:str=float_list_to_comma_separated_str(float_list=targets_list,digits=dig);
        print(f'targets_list: {targets_list}, type(targets_list): {type(targets_list)}, targets_list[0]: {targets_list[0]}, type(targets_list[0]): {type(targets_list[0])}');
        #Посмотреть, что будет без замены левых и правых квадратных скобок на пустоту
        #predictions_str=predictions_str.replace('[','').replace(']','');#7.8169,4.6257,1.108, вместо [7.8169],[4.6257],[1.108],
        #predictions_str:str=','.join(str(prediction)for prediction in targets_list);
        json_dict:dict={'predictions':predictions_str};
        json_filename:str='result_'+results_file_id+'.json';
        json_filename:str=f'result_{results_file_id}_{dig}_dig.json';
        with open(file=json_filename,mode='wt',encoding='UTF-8')as json_file:json.dump(obj=json_dict,fp=json_file);
    pipeline_ids_num:int=len(pipeline_ids);
    results_log_record:str=f"""
results_file_id: {results_file_id}
digits_round_min: {digits_round_min}, digits_round_max: {digits_round_max}
pipeline_ids_num: {pipeline_ids_num}
pipeline_ids_str: {pipeline_ids_str}
================""";
    with open(file='log_results.txt',mode='at',encoding='UTF-8')as results_lof_file:print(results_log_record,file=results_lof_file);
    print(f'Пайплайны с id {pipeline_ids_str} применены, их усреднённые результаты в файлах {tsv_filename} и {json_filename}');
    pass;

def create_coefs_and_bias_files(pipeline_ids:list[str],digits_round_min:int=2,digits_round_max:int=18)->None:
    """Функция принимает список id пайплайнов с одинаковыми значениями n_features_selected_randomly, randomly_selected_indexes, scaler (mean,var,
    scale) и создаёт txt файл, в который записывает усреднённые значения coef: array([...]) (массив из n_features_selected_randomly чисел
    типа float) и intercept.\n\nПонятно, что эта функция имеет смысл ТОЛЬКО в том случае, когда по условию задачи необходимо, чтобы
    модель была линейной. Вообще ансамблирование только линейных моделей всегда оставляет результирующую модель в классе линейных,
    поэтому если только условие об обязательной линейной модели не стоит, лучше НЕ использовать эту функцию"""
    pipelines_num:int=len(pipeline_ids);#Количество пайплайнов, по которым производится усреднение их coef и intercept
    #Считываем первый пайплайн для инициализации некоторых значений:
    print(f'ИНИЦИАЛИЗАЦИЯ НЕКОТОРЫХ ЗНАЧЕНИЙ ПЕРЕД СУММИРОВАНИЕМ coefs и bias:');
    pipeline_id:str=pipeline_ids[0];
    pkl_file_name:str=f'pipeline_{pipeline_id}.pkl';
    with open(file=pkl_file_name,mode='rb')as f:pipeline_dict:dict=pickle.load(file=f);
    print(f'pipeline_id: {pipeline_id}, pipeline_dict: {pipeline_dict}');
    n_features_selected_randomly:int=pipeline_dict['n_features_selected_randomly'];#n_features_selected_randomly одинаковое у всех пайплайнов
    randomly_selected_indexes:list[int]=pipeline_dict['randomly_selected_indexes'];#randomly_selected_indexes одинаковое у всех пайплайнов
    print(f'n_features_selected_randomly: {n_features_selected_randomly}, randomly_selected_indexes: {randomly_selected_indexes}');
    model=pipeline_dict['model'];
    scaler=pipeline_dict['scaler'];
    print(f'pipeline_id: {pipeline_id}');
    print(f'model.__dict__: {model.__dict__}');#model точно не None
    if scaler is not None:print(f'scaler.__dict__: {scaler.__dict__}');
    else:print(f'scaler: {scaler}');
    coefs_total:list[float]=[0.0 for i in range(n_features_selected_randomly)];#Инициализация массива весовых коэффициентов
    bias_total:float=0.0;#Инициализация смещения (intercept или bias) [все эти значения инициализируются нулями, так как a+0.0=a]
    
    #Считывание пайплайнов в цикле для суммирования соответствующих весовых коэффициентов coefs и смещения bias
    num_pipelines_added:int=0;
    num_pipelines_have_coef_and_intercept_attributes:int=0;
    for pipeline_id in pipeline_ids:
        pkl_file_name:str=f'pipeline_{pipeline_id}.pkl';
        print(f'Добавление пайплайна номер {num_pipelines_added+1} из {len(pipeline_ids)}, pipeline_id: {pipeline_id}, file_name: {pkl_file_name}');
        with open(file=pkl_file_name,mode='rb')as f:pipeline_dict:dict=pickle.load(file=f);
        #!!!Для пайплайнов scaler в общем случае не обязан быть одинаковым, поэтому skaler для каждого пайплайна считываем
        scaler_current=pipeline_dict['scaler'];#заново (из обрабатываемого в данный момент пайплайна)
        
        scaler_mean_list_float:list[float]=[float(num)for num in scaler_current.__dict__['mean_']];
        scaler_var_list_float:list[float]=[float(num)for num in scaler_current.__dict__['var_']];
        scaler_scale_list_float:list[float]=[float(num)for num in scaler_current.__dict__['scale_']];
        scaler_scale_sqr_list_float:list[float]=[num**2 for num in scaler_scale_list_float];#Для проверки того, что var - это дисперсия,
        #а scale - это квадратный корень из дисперсии (или 1.0, если дисперсия равна 0.0)

        model_current=pipeline_dict['model'];
        coefs_current:np.ndarray=np.zeros(shape=(n_features_selected_randomly),dtype=np.float64);#Инициализация массива текущих считанных из
        #pkl файла коэффициентов (на тот случай, если такого атрибута нет)
        bias_current:np.ndarray=np.zeros(shape=(1),dtype=np.float64);#Инициализация текущего значения считанного из
        #pkl файла bias (на тот случай, если такого атрибута нет)
        #Считывание coef и intercept, если они оба есть у модели текущего пайплайна
        #ПОКА ЧТО СЧИТАЕМ, ЧТО ВСЕ scaler - это StandardScaler (если scaler_current - это НЕ StandardScaler, то пропускаем этот пайплайн)
        if (hasattr(model_current,'coef_'))and(hasattr(model_current,'intercept_'))and(isinstance(scaler_current,StandardScaler)==True):
            num_pipelines_have_coef_and_intercept_attributes=num_pipelines_have_coef_and_intercept_attributes+1;
            coefs_current:np.ndarray=model_current.__dict__['coef_'];
            bias_current:np.ndarray=model_current.__dict__['intercept_'];
            print(f'pipeline_id: {pipeline_id}, coefs_current: {coefs_current}, bias_current: {bias_current}');
            print(f'coefs_current: {coefs_current}, bias_current: {bias_current}');
            print(f'type(coefs_current): {type(coefs_current)}, type(bias_current): {type(bias_current)}');
            if len(coefs_current.shape)==1:#Если coefs_current: [6.77108822e-02 8.84446049e-01 3.14625849e+01 0.00000000e+00]
                for i in range(n_features_selected_randomly):coefs_total[i]=coefs_total[i]+float(coefs_current[i]);
            elif len(coefs_current.shape)==2:#Если coefs_current: [[ 0.00489083  0.78579048  0.         -0.23675243  0.79130886]]
                for i in range(n_features_selected_randomly):coefs_total[i]=coefs_total[i]+float(coefs_current[0][i]);
            if type(bias_current)==np.ndarray:bias_total=bias_total+float(bias_current[0]);
            elif type(bias_current)==np.float64:bias_total=bias_total+float(bias_current);
            elif type(bias_current)==float:bias_total=bias_total+bias_current;
            else:print(f'bias_current: {bias_current}, type(bias_current): {type(bias_current)}, значение bias_current НЕ ДОБАВЛЕНО, так как его тип не предусмотрен в коле функции, НЕОБХОДИМО предусмотреть обработку случая с этим типом');
        else:
            print(f"hasattr(model_current,'coef_'): {hasattr(model_current,'coef_')}, hasattr(model_current,'intercept_'): {hasattr(model_current,'intercept_')}, эта модель (модель из этого пайплайна) НЕ добавлена, так как у неё нет хотя бы одного из нужных атрибутов");
        num_pipelines_added=num_pipelines_added+1;
    print(f'Все пайплайны обработаны, из имеющих атрибуты coef_ и intercept_ в своих моделях (таких {num_pipelines_have_coef_and_intercept_attributes} из {pipelines_num} или {(100*num_pipelines_have_coef_and_intercept_attributes/pipelines_num):.4f}%) добавлены коэффициенты и смещение');
    for i in range(n_features_selected_randomly):coefs_total[i]=coefs_total[i]/num_pipelines_have_coef_and_intercept_attributes;
    bias_total=bias_total/num_pipelines_have_coef_and_intercept_attributes;
    print(f'После усреднения по {num_pipelines_have_coef_and_intercept_attributes} пайплайнам (до учёта scaler):');
    print(f'coefs: {coefs_total}, bias: {bias_total}');
    str_list_to_txt:list[str]=[];
    pipeline_ids_str:str=' '.join(pipeline_ids);
    str_list_to_txt.append(f'n_features_selected_randomly: {n_features_selected_randomly}, pipelines_num: {pipelines_num}, pipeline_ids_str: {pipeline_ids_str}\n');
    str_list_to_txt.append(f'{num_pipelines_have_coef_and_intercept_attributes} из {pipelines_num} пайплайнов (или {(100*num_pipelines_have_coef_and_intercept_attributes/pipelines_num):.4f}%) имеют атрибуты coef_ и intercept_, по этим пайплайнам выполняется усреднение\n');
    str_list_to_txt.append(f'После усреднения по {num_pipelines_have_coef_and_intercept_attributes} пайплайнам (до учёта scaler):\n');
    str_list_to_txt.append(f'coefs: {coefs_total}, bias: {bias_total}\n');
    str_list_to_txt.append(f'scaler_mean_list_float: {scaler_mean_list_float}\n');
    str_list_to_txt.append(f'scaler_var_list_float: {scaler_var_list_float}\n');
    str_list_to_txt.append(f'scaler_scale_list_float: {scaler_scale_list_float}\n');
    str_list_to_txt.append(f'scaler_scale_sqr_list_float: {scaler_scale_sqr_list_float}\n');
    
    
    coefs_after_scaler:list[float]=[0.0]*n_features_selected_randomly;
    bias_after_scaler:float=bias_total;
    for i in range(n_features_selected_randomly):
        #Каждый коэффициент делится на соответствующий scale
        coefs_after_scaler[i]=coefs_total[i]/scaler_scale_list_float[i] if scaler_scale_list_float[i]!=0.0 else 0.0;
        #Вычитаем вклад mean из bias
        bias_after_scaler=bias_after_scaler-coefs_after_scaler[i]*scaler_mean_list_float[i];
    str_list_to_txt.append(f'После усреднения по {num_pipelines_have_coef_and_intercept_attributes} пайплайнам (после учёта scaler):\n');
    print(f'coefs: {coefs_after_scaler}, bias: {bias_after_scaler}');
    str_list_to_txt.append(f'coefs: {coefs_after_scaler}, bias: {bias_after_scaler}\n');


    str_list_to_txt.append(f'================================\n');
    with open(file='log_results.txt',mode='at',encoding='UTF-8')as f_log:f_log.writelines(str_list_to_txt);
    pass;

#Действия, выполняемые перед каждым запуском:
opened_data_all_features,opened_target,opened_ids,closed_data_all_features,closed_target,closed_ids=load_data_from_npy(rewrite_features_csv_files=False);
create_log_files();

#Основной цикл программы:
command_num:int=0;
while command_num>-1:
    print(f'=====================================');
    print(f'1 => выполнить создание, кросс-валидацию и проверку на holdout одного пайплайна со случайными гиперпараметрами n раз (функция run_one_pipeline_experiment_v1)');
    print(f'2 => создать json и tsv файлы с предсказанием пайплайна или средним предсказанием нескольких пайплайнов из списка их id');
    print(f'3 => анализ csv лога с результатами пайплайнов (для отбора id лучших пайплайнов)');
    print(f'4 => анализ txt лога с результатами пайплайнов (для отбора индексов наиболее часто случайно выбираемых в лучших пайплайнах признаков)');
    print(f'5 => вывод информации о содержимом одного pkl файла');
    print(f'6 => создать txt файл со значениями coef и bias для пайплайна или усреднёнными значениями нескольких пайплайнов из списка их id');

    print(f'-1 => выйти из программы');
    print(f'=====================================');

    input_str:str=input('Введите номер команды: ');
    print(f'Введено: {input_str}');
    command_num=int(input_str);
    if command_num==1:#1 => выполнить кросс-валидацию с проверкой на holdout n раз (фунция run_one_pipeline_experiment_v1)
        num_of_experiments:int=int(input('Введите количество экспериментов: '));
        num_features_select_from_all_min:int=0;num_features_select_from_all_max:int=0;
        randomly_selected_indexes_str:str=input('Введите список номеров признаков для отбора через ПРОБЕЛ (список для отбора одинакового набора признаков во всех экспериментах или просто Enter для случайного задания списка в каждом эксперименте), пример списка (без квадратных скобок): [218 307 56 266 63 67 77 336 105 376 59 73 257 42]: ');
        #indexes_times_list_dicts (after sorting): [{'index': 218, 'times': 114}, {'index': 307, 'times': 113}, {'index': 56, 'times': 111}, {'index': 266, 'times': 111}, {'index': 63, 'times': 110}, {'index': 67, 'times': 110}, {'index': 77, 'times': 109}, {'index': 336, 'times': 108}, {'index': 84, 'times': 107}, {'index': 105, 'times': 107}, {'index': 376, 'times': 106}, {'index': 59, 'times': 105}, {'index': 73, 'times': 105}, {'index': 257, 'times': 105}, {'index': 42, 'times': 103}, {'index': 362, 'times': 103}, ...
        #14 наиболее часто использованных признаков (в виде списка): [218, 307, 56, 266, 63, 67, 77, 336, 84, 105, 376, 59, 73, 257]
        #Восьмым (считая с нуля) в этом списке является признак номер 84, то есть это признак X84 из файла "all_features_opened_data.csv". Все значения этого признака равны нулю. Вместо него используем следующий признак (следующий за признаком 257), это признак {'index': 42, 'times': 103}. Этот признак подходит, у него все 800 значений у открытых данных уникальны (то есть его дисперсия не равна нулю). Вместо списка [218, 307, 56, 266, 63, 67, 77, 336, 84, 105, 376, 59, 73, 257] испоьзуем список [218, 307, 56, 266, 63, 67, 77, 336, 105, 376, 59, 73, 257, 42].
        if len(randomly_selected_indexes_str)==0:#Если список индексов не задан, то они могут выбираться случайно (если их количество равно 0)
            randomly_selected_indexes:list[int]=None;
            num_features_select_from_all_min:int=int(input('Введите минимальное количество случайно отбираемых признаков в каждом эксперименте (или 0 для использования всех признаков): '));
            num_features_select_from_all_max:int=int(input('Введите максимальное количество случайно отбираемых признаков в каждом эксперименте (или 0 для использования всех признаков): '));
        else:randomly_selected_indexes:list[int]=[int(num)for num in randomly_selected_indexes_str.split(sep=' ')];
        use_only_linear_models_str:str=input(f'Использовать только полностью линейные модели (имеющие атрибуты coef_ и intercept_) [1=True, 0=False, ничего=False]: ');
        if use_only_linear_models_str in['','0','False','false','F','f']:use_only_linear_models:bool=False;
        elif use_only_linear_models_str in['1','True','true','T','t']:use_only_linear_models:bool=True;
        use_imputer_probability:float=str_to_float(s=input(f'Введите вероятность использования imputer в каждом эксперименте (от 0.0 до 1.0) [если 0.0 то imputer никогда не используется, если 1.0 то imputer используется в каждом эксперименте, если 0.5 то imputer используется примерно в половине экспериментов]: '),num_min=0.0,num_max=1.0,num_default=0.9);
        use_var_thresholder_probability:float=str_to_float(s=input(f'Введите вероятность использования var_thresholder в каждом эксперименте (от 0.0 до 1.0) [если 0.0 то var_thresholder никогда не используется, если 1.0 то var_thresholder используется в каждом эксперименте, если 0.5 то var_thresholder используется примерно в половине экспериментов]: '),num_min=0.0,num_max=1.0,num_default=0.9);
        use_scaler_probability:float=str_to_float(s=input(f'Введите вероятность использования scaler в каждом эксперименте (от 0.0 до 1.0) [если 0.0 то scaler никогда не используется, если 1.0 то scaler используется в каждом эксперименте, если 0.5 то scaler используется примерно в половине экспериментов]: '),num_min=0.0,num_max=1.0,num_default=0.95);
        use_feature_selector_probability:float=str_to_float(s=input(f'Введите вероятность использования feature_selector в каждом эксперименте (от 0.0 до 1.0) [если 0.0 то feature_selector никогда не используется, если 1.0 то feature_selector используется в каждом эксперименте, если 0.5 то feature_selector используется примерно в половине экспериментов]: '),num_min=0.0,num_max=1.0,num_default=0.9);
        prefered_scaler_types_str:str=input(f'Введите строку с предпочтительными типами scaler (разделяя их через пробел) [все типы scaler: MaxAbsScaler MinMaxScaler RobustScaler StandardScaler] или просто Enter чтобы не выбирать предпочтительные типы scaler: ');
        prefered_feature_selector_types_str:str=input(f'Введите строку с предпочтительными типами feature_selector (разделяя их через пробел) [все типы feature_selector: GenericUnivariateSelect SelectFdr SelectFpr SelectFwe SelectKBest SelectPercentile SelectFromModel RFE RFECV SequentialFeatureSelector] или просто Enter чтобы не выбирать предпочтительные типы feature_selector: ');
        prefered_fs_estimator_types_str:str=input(f'Введите строку с предпочтительными типами fs_estimator (разделяя их через пробел) [все типы fs_estimator: LogisticRegression PassiveAggressiveClassifier Perceptron RidgeClassifier SGDClassifier LinearRegression Ridge SGDRegressor ElasticNet Lars Lasso LassoLars LassoLarsIC OrthogonalMatchingPursuit ARDRegression BayesianRidge HuberRegressor QuantileRegressor RANSACRegressor TheilSenRegressor GammaRegressor PoissonRegressor TweedieRegressor PassiveAggressiveRegressor] или просто Enter чтобы не выбирать предпочтительные типы fs_estimator: ');
        prefered_model_types_str:str=input(f'Введите строку с предпочтительными типами model (разделяя их через пробел) [все типы model: AdaBoostClassifier BaggingClassifier ExtraTreesClassifier GradientBoostingClassifier HistGradientBoostingClassifier RandomForestClassifier XGBClassifier LGBMClassifier LogisticRegression PassiveAggressiveClassifier Perceptron RidgeClassifier SGDClassifier LinearRegression Ridge SGDRegressor ElasticNet Lars Lasso LassoLars LassoLarsIC ARDRegression BayesianRidge HuberRegressor QuantileRegressor RANSACRegressor TheilSenRegressor GammaRegressor PoissonRegressor TweedieRegressor PassiveAggressiveRegressor AdaBoostRegressor BaggingRegressor ExtraTreesRegressor GradientBoostingRegressor HistGradientBoostingRegressor RandomForestRegressor] или просто Enter чтобы не выбирать предпочтительные типы model: ');
        if prefered_scaler_types_str=='':prefered_scaler_types:list[str]=None;
        else:prefered_scaler_types:list[str]=prefered_scaler_types_str.replace('  ',' ').split(sep=' ');
        if prefered_feature_selector_types_str=='':prefered_feature_selector_types:list[str]=None;
        else:prefered_feature_selector_types:list[str]=prefered_feature_selector_types_str.replace('  ',' ').split(sep=' ');
        if prefered_fs_estimator_types_str=='':prefered_fs_estimator_types:list[str]=None;
        else:prefered_fs_estimator_types:list[str]=prefered_fs_estimator_types_str.replace('  ',' ').split(sep=' ');
        if prefered_model_types_str=='':prefered_model_types:list[str]=None;
        else:prefered_model_types:list[str]=prefered_model_types_str.replace('  ',' ').split(sep=' ');
        n_cpu_cores:int=int(input(f'Введите количество потоков процессора для использования в обучении некоторых моделей (-1 для использования всех потоков, -2 для использования всех потоков кроме одного и т.д.): '));


        for i in range(num_of_experiments):
            try:
                print(f'Эксперимент {i+1}/{num_of_experiments}... ',end='');
                pipeline_id:str=run_one_pipeline_experiment_v1(num_features_select_from_all_min=num_features_select_from_all_min,num_features_select_from_all_max=num_features_select_from_all_max,randomly_selected_indexes=randomly_selected_indexes,problem_type='regression',task_output='mono_output',score_type='mean_squared_error',
                use_imputer_probability=use_imputer_probability,use_var_thresholder_probability=use_var_thresholder_probability,use_scaler_probability=use_scaler_probability,prefered_scaler_types=prefered_scaler_types,use_feature_selector_probability=use_feature_selector_probability,prefered_feature_selector_types=prefered_feature_selector_types,prefered_fs_estimator_types=prefered_fs_estimator_types,scaler_type=None,scaler_hyperparams=None,model_type=None,prefered_model_types=prefered_model_types,model_hyperparams=None,num_folds=10,score_valid_min_threshold=None,score_valid_max_threshold=0.10,use_only_linear_models=use_only_linear_models,n_cpu_cores=n_cpu_cores);
            except Exception as ex:
                print(f'Возникло исключение, type(ex): {type(ex)}, ex: {ex}');
    elif command_num==2:#2 => создать json и tsv файлы с предсказанием пайплайна или средним предсказанием нескольких пайплайнов из списка их id
        pipeline_ids_str:str=input('Введите id пайплайна или нескольких пайплайнов через запятую или пробел (например, [08JZRAWXBE5N43MX] или [08JZRAWXBE5N43MX,2352C29OXLDYGPAL,J0KZOWU71FHE3TCR,EENT8VMHI4CK4D24]) (БЕЗ КВАДРАТНЫХ СКОБОК): ');
        digits_round_min,digits_round_max=[int(num_s)for num_s in input('Введите минимальное и максимальное количество цифр округления (например, 2 18): ').split(sep=' ')]
        if ','in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep=',');
        elif ' 'in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep=' ');
        elif '/'in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep='/');
        elif '|'in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep='|');
        create_predictions_files(pipeline_ids=pipeline_ids_list,digits_round_min=digits_round_min,digits_round_max=digits_round_max);
    elif command_num==3:#анализ csv лога с результатами пайплайнов
        score_valid_mean_threshold_min:float=0.0;
        score_valid_mean_threshold_max:float=0.090;
        score_test_threshold_min:float=0.0;
        score_test_threshold_max:float=0.115;
        n_features_selected_randomly_threshold_min:float=None;
        n_features_selected_randomly_threshold_max:float=None;
        log_pipelines_csv_file_name:str=input(f'Введите название csv файла с логом, для которого нужно выполнить анализ (например, log_pipelines.csv) [для анализа файла с именно этим названием можно просто нажать Enter]: ');
        if len(log_pipelines_csv_file_name)==0:log_pipelines_csv_file_name='log_pipelines.csv';
        analize_log_pipelines_csv(log_pipelines_csv_file_name=log_pipelines_csv_file_name,score_valid_mean_threshold_min=score_valid_mean_threshold_min,score_valid_mean_threshold_max=score_valid_mean_threshold_max,score_test_threshold_min=score_test_threshold_min,score_test_threshold_max=score_test_threshold_max,n_features_selected_randomly_threshold_min=n_features_selected_randomly_threshold_min,n_features_selected_randomly_threshold_max=n_features_selected_randomly_threshold_max);
        pass;
    elif command_num==4:#анализ txt лога с результатами пайплайнов
        log_pipelines_txt_file_name:str=input(f'Введите название txt файла с логом, для которого нужно выполнить анализ (например, log_pipelines.txt) [для анализа файла с именно этим названием можно просто нажать Enter]: ');
        if len(log_pipelines_txt_file_name)==0:log_pipelines_txt_file_name='log_pipelines.txt';
        analize_log_pipelines_txt(log_pipelines_txt_file_name=log_pipelines_txt_file_name);

        pass;
    elif command_num==5:
        pkl_file_name:str=input('Введите название pkl файла (например, pipeline_RM3W9PGWI65QRNXI.pkl) или просто id пайплайна (например, RM3W9PGWI65QRNXI): ');
        if 'pipeline_'not in pkl_file_name:pkl_file_name=f'pipeline_{pkl_file_name}.pkl';
        analize_one_pkl_file(pkl_file_name=pkl_file_name);

        pass;
    elif command_num==6:#6 => создать txt файл со значениями coef и bias для модели или усреднёнными значениями нескольких пайплайнов из списка их id
        pipeline_ids_str:str=input('Введите id пайплайна или нескольких пайплайнов через запятую или пробел (например, [93P121PG8FACD4L2] или [93P121PG8FACD4L2 0UVA9BKRDG2E7EX6 0271ZHM0Z5R3HXLS 04HVI5VS4FKMIVBG 05DZJE2OQWUXM93R 06SJSYUV7IEKGEXX]) (БЕЗ КВАДРАТНЫХ СКОБОК): ');
        digits_round_min,digits_round_max=[int(num_s)for num_s in input('Введите минимальное и максимальное количество цифр округления (например, 2 18): ').split(sep=' ')]
        if ','in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep=',');
        elif ' 'in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep=' ');
        elif '/'in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep='/');
        elif '|'in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep='|');
        create_coefs_and_bias_files(pipeline_ids=pipeline_ids_list,digits_round_min=digits_round_min,digits_round_max=digits_round_max);

print(f'Работа программы завершена');



