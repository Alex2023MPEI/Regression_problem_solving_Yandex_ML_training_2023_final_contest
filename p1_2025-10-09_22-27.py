#Ссылка на контест для этой задачи (assignment_final): https://contest.yandex.ru/contest/56809/problems/
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

def analize_log_models_csv(log_models_csv_file_name:str='log_models.csv',score_valid_mean_threshold_min:float=None,score_valid_mean_threshold_max:float=None,score_test_threshold_min:float=None,score_test_threshold_max:float=None,n_features_selected_threshold_min:float=None,n_features_selected_threshold_max:float=None)->None:
    log_models_dicts_list:list[dict]=read_csv(filename=log_models_csv_file_name,delimiter_values=',');
    if'n_features_all'in log_models_dicts_list[0].keys():
        for model_dict in log_models_dicts_list:model_dict['n_features_all']=int(model_dict['n_features_all']);
    if'n_features_selected'in log_models_dicts_list[0].keys():
        for model_dict in log_models_dicts_list:model_dict['n_features_selected']=int(model_dict['n_features_selected']);
    if'score_valid_mean'in log_models_dicts_list[0].keys():
        for model_dict in log_models_dicts_list:model_dict['score_valid_mean']=float(model_dict['score_valid_mean']);
    if'score_test'in log_models_dicts_list[0].keys():
        for model_dict in log_models_dicts_list:model_dict['score_test']=float(model_dict['score_test']);
    print(f'log_models_dicts_list: {log_models_dicts_list}');
    print(f'len(log_models_dicts_list): {len(log_models_dicts_list)}');
    numeric_keys:list[str]=['n_features_all','n_features_selected','score_valid_mean','score_test'];
    numeric_keys_stats:list[dict[str:float]]=[];
    for i in range(len(numeric_keys)):
        numeric_keys_stats.append({});
        name:str=numeric_keys[i];
        numeric_keys_stats[i]['name']=numeric_keys[i];
        numeric_keys_stats[i]['number']=sum([1 for j in range(len(log_models_dicts_list))]);
        numeric_keys_stats[i]['sum']=sum([log_models_dicts_list[j][name]for j in range(len(log_models_dicts_list))]);
        numeric_keys_stats[i]['mean']=numeric_keys_stats[i]['sum']/numeric_keys_stats[i]['number'];
        numeric_keys_stats[i]['min']=min([log_models_dicts_list[j][name]for j in range(len(log_models_dicts_list))]);
        numeric_keys_stats[i]['max']=max([log_models_dicts_list[j][name]for j in range(len(log_models_dicts_list))]);
        numeric_keys_stats[i]['std']=(sum([(log_models_dicts_list[j][name]-numeric_keys_stats[i]['mean'])**2 for j in range(len(log_models_dicts_list))])/numeric_keys_stats[i]['number'])**0.5;
        print(f'i: {i}, numeric_keys_stats[i]: {numeric_keys_stats[i]}');
        pass;
    for name in ['n_features_selected','score_valid_mean','score_test']:
        min_sub_list:list[dict]=sorted(log_models_dicts_list,key=lambda d:d[name],reverse=False)[:10];
        print(f'Модели с наименьшими значениями {name}:');
        for i in range(len(min_sub_list)):print(f'{i}) {min_sub_list[i]}');
        max_sub_list:list[dict]=sorted(log_models_dicts_list,key=lambda d:d[name],reverse=True)[:10];
        print(f'Модели с наибольшими значениями {name}:');
        for i in range(len(max_sub_list)):print(f'{i}) {max_sub_list[i]}');
    s_lst:list[dict]=[d for d in log_models_dicts_list];#s_lst - список только тех моделей, у которых числовые значения удовлетворяют ограничениям
    if score_valid_mean_threshold_min is not None:s_lst:list[dict]=[d for d in s_lst if d['score_valid_mean']>=score_valid_mean_threshold_min];
    if score_valid_mean_threshold_max is not None:s_lst:list[dict]=[d for d in s_lst if d['score_valid_mean']<=score_valid_mean_threshold_max];
    if score_test_threshold_min is not None:s_lst:list[dict]=[d for d in s_lst if d['score_test']>=score_test_threshold_min];
    if score_test_threshold_max is not None:s_lst:list[dict]=[d for d in s_lst if d['score_test']<=score_test_threshold_max];
    if n_features_selected_threshold_min is not None:s_lst:list[dict]=[d for d in s_lst if d['n_features_selected']>=n_features_selected_threshold_min];
    if n_features_selected_threshold_max is not None:s_lst:list[dict]=[d for d in s_lst if d['n_features_selected']<=n_features_selected_threshold_max];
    print(f'Список отобранных моделей:');
    for d in s_lst:print(d);
    print(f'В списке отобранных моделей {len(s_lst)} моделей из {len(log_models_dicts_list)} ({(100*len(s_lst)/len(log_models_dicts_list)):.4f}%)');
    print(f'Модели отобраны с ограничениями: {locals()}');
    ids_of_selected_models:list[str]=sorted([d['model_id']for d in s_lst]);
    print(f'id выбранных {len(ids_of_selected_models)} моделей: [{" ".join(ids_of_selected_models)}]');
    pass;

def analize_log_models_txt(log_models_txt_file_name:str='log_models.txt',num_of_most_times_used_features_indexes:int=14)->None:
    """Функция проводит анализ файла log_models.txt и выявляет, какие признаки (по индексам) наиболее часто использовались в лучших
    моделях. Это полезно в том случае, если необходимо построить модель с использованием не более чем некоторого количества признаков
    (например, в задаче [B. Финальное соревнование: задача 2], где в условии сказано: [Вторая модель должна быть линейной, т.е.
    представлять собой линейную комбинацию признаков плюс смещение, модель не должна использовать более 15 параметров (14 весов плюс
    смещение)])\n
    При таком условии:
    1. Определяем лучшие признаки (условно, если признак номер 15 использован в 100 лучших моделях, а признак номер 23 использован в
    60 лучших моделях, то наверное признак номер 15 полезнее, чем признак номер 23). Для каждого признака (его индекса) опредеяем,
    количество раз, сколько этот признак использован в лучших моделях, затем сортируем по убыванию этих количеств и отбираем 14 тех
    признаков, которые использованы в наибольшем количестве моделей (именно 14 признаков, так как установлено ограничение в 15
    параметров, один из которых - это смещение [bias])
    2. Выполняем построение моделей с отбором конкретно этих 14 признаков (использование различных моделей, RandomSearch для подбора
    гиперпараметров, кросс-валидация на 10 фолдов, тестирование на отложенной выборке, обучение прошедших порог моделей на всех
    открытых данных, их сохранение в *.pkl файлы вместе со Scaler и запись результатов в txt и csv логи)
    3. Затем отбор моделей, удоветворяющих пороговым значениям score_valid_mean и score_test (функция analize_log_models_csv) и
    усреднение их предсказаний. Каждая модель линейная и имеет 14 коэффициентов k0,...,k13 + bias следовательно для усреднения
    их предсказаний усредняем их коэффициенты k0,...,k13 и bias. Например, если отобрано 100 лучших моделей, значит
    k0=(k0[0]+k0[1]+..+k0[99])/100, k1=(k1[0]+k1[1]+..+k1[99])/100, ..., k13=(k13[0]+k13[1]+..+k13[99])/100,
    bias=(bias[0]+bias[1]+..+bias[99])/100
    Нужно будет ещё подумать над scaler для каждого из 14 признаков, но тут скорее всего Scaler у всех моделей будет одинаковый
    (так как Scaler не зависит от того, какая после него применена модель + каждая итоговая модель [сохраняемая затем в *.pkl файл]
    обучается на всех 100% открытых данных, поэтому очевидно, что при задании фиксированного списка лучших признаков [selected_indexes]
    scaler у моделей во всех *.pkl файлах должны быть одинаковые).\n
    После применения функции analize_one_pkl_file к двум pkl файлам моделей подтверждено, что всё содержимое scaler у разных моделей
    полностью совпадает, например: value.__dict__: {'with_mean': True, 'with_std': True, 'copy': True, 'n_features_in_': 392, 'n_samples_seen_': 800, 'mean_': array([-5.62410914e-01,  2.58281096e-01, -5.68630481e-01, -6.69724427e-04,...
    \n
    Равенство Scaler для разных моделей - это правильно, но почему-то и model и scaler выдают: 'n_features_in_': 392
    \nP.S. 'n_features_in_': 392 - это было следствием ошибки в начале код функции run_one_model_experiment_v5 в части: 
    отбор num_features_select_from_all признаков из всех. Из-за этой ошибки если максимальное и минимальное количества используемых
    признаков заданы как нули, то даже при заданном списке индексов в массивы opened_data и closed_data попадали все признаки (392
    признака в этой задаче). Именно из-за такого огромного количества признаков при проведении эксперимента много раз за почти сутки
    появилось только около 40 +-хороших моделей. Теперь эта ошибка ИСПРАВЛЕНА. Теперь сначала проверяется наличие списка индексов
    признаков, затем если его нет, то список индексов признаков заполняется.

    \nЕсли ограниечение на количество признаков не установлено, то вероятно эта функция не очень нужна"""
    with open(file=log_models_txt_file_name,mode='rt',encoding='UTF-8')as f:
        txt_log_lines:list[str]=[line.rstrip('\n') for line in f.readlines()];
    print(f'len(txt_log_lines): {len(txt_log_lines)}');
    print(f'txt_log_lines[4]: {txt_log_lines[4]}');
    selected_indexes_lines:list[str]=[line for line in txt_log_lines if 'selected_indexes: 'in line];
    selected_indexes_lines=[line.replace('selected_indexes: [','').replace(']',',').replace(' ','')for line in selected_indexes_lines];
    print(f'len(selected_indexes_lines): {len(selected_indexes_lines)}');
    print(f'selected_indexes_lines[:10]: {selected_indexes_lines[:10]}');
    n_features_selected_lines:list[str]=[line for line in txt_log_lines if 'n_features_selected: 'in line];
    n_features_selected_lines=[line.replace('n_features_selected: ','') for line in n_features_selected_lines];
    n_features_selected_ints:list[int]=[int(line)for line in n_features_selected_lines];
    print(f'len(n_features_selected_ints): {len(n_features_selected_ints)}');
    print(f'n_features_selected_ints[:50]: {n_features_selected_ints[:50]}');
    print(f'sum(n_features_selected_ints): {sum(n_features_selected_ints)}');
    selected_indexes_str:str=''.join(selected_indexes_lines);
    selected_indexes_list:list[int]=[int(num)for num in selected_indexes_str.split(sep=',')if len(num)>0];#Чтобы не пытаться
    #преобразовать пустую строку после последней запятой в число типа int
    print(f'len(selected_indexes_list): {len(selected_indexes_list)}');
    print(f'selected_indexes_list[:50]: {selected_indexes_list[:50]}');
    if sum(n_features_selected_ints)==len(selected_indexes_list):
        print(f'sum(n_features_selected_ints): {sum(n_features_selected_ints)}, len(selected_indexes_list): {len(selected_indexes_list)}, эти числа равны, проверка работает правильно');
    else:#Сумма n_features_selected_ints должна быть равна длине selected_indexes_list (и равна суммарному количеству выбранных признаков во всех моделях, попавших в файл log_models.txt)
        print(f'sum(n_features_selected_ints): {sum(n_features_selected_ints)}, len(selected_indexes_list): {len(selected_indexes_list)}, эти числа НЕ равны, проверка показывает наличие ошибки');
    selected_non_zero_times_indexes:list[int]=sorted(list(set(selected_indexes_list)));
    print(f'selected_non_zero_times_indexes: {selected_non_zero_times_indexes}, len(selected_non_zero_times_indexes): {len(selected_non_zero_times_indexes)}');
    #Сохранение в словарь {index:num_of_this_index}
    indexes_times_dict:dict[int:int]={};#Ключи - индексы, значения - их количества
    for ind in selected_non_zero_times_indexes:indexes_times_dict[ind]=0;
    for i in selected_indexes_list:indexes_times_dict[i]=indexes_times_dict[i]+1;
    print(f'indexes_times_dict: {indexes_times_dict}');
    #Сохранение в список словарей [{'index':,'times':},...,{'index':,'times':}]
    indexes_times_list_dicts:list[dict[str:int]]=[];
    for ind in selected_non_zero_times_indexes:indexes_times_list_dicts.append({'index':ind,'times':0});
    #Это рабочий вариант, но лучше так не делать, так как он рассчитывает на то, что словари для всех выбранных индексов
    #расположены по порядку увеличения этих индексов без пропусков
    #for ind in selected_indexes_list:indexes_times_list_dicts[ind]['times']=indexes_times_list_dicts[ind]['times']+1;
    #Другой вариант (тоже рабочий, но должен быть более общим):
    for ind in selected_indexes_list:
        index:int=None;
        for index_key in range(len(indexes_times_list_dicts)):#Для эффективности можно заменить это for на while но число признаков
            if indexes_times_list_dicts[index_key]['index']==ind:#вряд ли будет больше ста тысяч, поэтому можно так оставить
                index=ind;
        indexes_times_list_dicts[index]['times']=indexes_times_list_dicts[index]['times']+1;
    print(f'indexes_times_list_dicts (before sorting): {indexes_times_list_dicts}');
    indexes_times_list_dicts.sort(key=lambda d:d['times'],reverse=True);#Сортировка по убыванию количеств
    print(f'indexes_times_list_dicts (after sorting): {indexes_times_list_dicts}');
    #[{'index':218,'times':114},{'index':307,'times':113},...,{'index':290,'times':71},{'index':269,'times': 69}]
    print(f'{num_of_most_times_used_features_indexes} наиболее часто использованных признаков (в виде словаря номер:количество): {indexes_times_list_dicts[:num_of_most_times_used_features_indexes]}');
    print(f'{num_of_most_times_used_features_indexes} наиболее часто использованных признаков (в виде списка): {[d["index"] for d in indexes_times_list_dicts[:num_of_most_times_used_features_indexes]]}');
    #14 наиболее часто использованных признаков (в виде словаря номер:количество): [{'index': 218, 'times': 114}, {'index': 307, 'times': 113}, {'index': 56, 'times': 111}, {'index': 266, 'times': 111}, {'index': 63, 'times': 110}, {'index': 67, 'times': 110}, {'index': 77, 'times': 109}, {'index': 336, 'times': 108}, {'index': 84, 'times': 107}, {'index': 105, 'times': 107}, {'index': 376, 'times': 106}, {'index': 59, 'times': 105}, {'index': 73, 'times': 105}, {'index': 257, 'times': 105}]
    #14 наиболее часто использованных признаков (в виде списка): [218, 307, 56, 266, 63, 67, 77, 336, 84, 105, 376, 59, 73, 257]
    
    pass;

def analize_one_pkl_file(pkl_file_name:str)->None:
    """Функция выводит информацию о содержимом одного pkl файла"""
    with open(file=pkl_file_name,mode='rb')as pkl_file:#binary mode doesn't take an encoding argument
        print(f'======== Информация об одном pkl файле ========:');
        pkl_file_size:int=os.path.getsize(filename=pkl_file_name);
        print(f'pkl_file_name: {pkl_file_name}, pkl_file_size: {pkl_file_size}');        
        pkl_obj=pickle.load(file=pkl_file);
        print(f'pkl_obj: {pkl_obj}');
        print(f'type(pkl_obj): {type(pkl_obj)}');
        if type(pkl_obj)==list:
            pass;
        elif type(pkl_obj)==dict:
            pkl_obj_keys=pkl_obj.keys();
            print(f'pkl_obj_keys: {pkl_obj_keys}');
            for key in pkl_obj_keys:
                value=pkl_obj[key];
                __dict__str:str=f'Объект {value} не имеет атрибута __dict__';
                if hasattr(value,'__dict__'):__dict__str=f'value.__dict__: {value.__dict__}';
                print(f'key: {key}, type(key): {type(key)}, value: {value}, type(value): {type(value)}, \n value.__dir__(): {value.__dir__()}, \n {__dict__str}\n');
            pass;


        pass;
    pass;

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


def load_data_from_npy()->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Загрузка данных из *.npy файлов (эти файлы содержат массивы NumPy)"""
    assert os.path.exists('hw_final_open_data.npy'),'Please, download hw_final_open_data.npy and place it in the working directory'
    assert os.path.exists('hw_final_open_target.npy'),'Please, download hw_final_open_target.npy and place it in the working directory'
    assert os.path.exists('hw_final_closed_data.npy'),'Please, download hw_final_closed_data.npy and place it in the working directory'
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
    #В программе для задачи F. Биометрия было так (13936 образцов открытых данных, 3413 образцов закрытых данных, 1536 признаков):
    #x.shape: (13936, 1536), y.shape: (13936,), ids.shape: (13936,),
    #x.shape: (3413, 1536), y.shape: (3413,), ids.shape: (3413,),

    return opened_data_all_features,opened_target,opened_ids,closed_data_all_features,closed_target,closed_ids;

def create_log_files()->None:
    """Функция создаёт log файлы (если они не существуют)"""
    if pathlib.Path('log_models.txt').exists()==False:#Создать файл log_models.txt если его не существует
        with open(file='log_models.txt',mode='wt',encoding='UTF-8')as f_log:
            pass;
    if pathlib.Path('log_models.csv').exists()==False:#Создать файл log_models.csv если его не существует и заполнить его заголовок
        with open(file='log_models.csv',mode='wt',encoding='UTF-8')as f_log:
            header_str:str=f'model_id,n_features_all,n_features_selected,model_type,score_type,score_valid_mean,score_test';
            print(header_str,file=f_log);
            pass;
    if pathlib.Path('log_results.txt').exists()==False:#Создать файл log_results.txt если его не существует
        with open(file='log_results.txt',mode='wt',encoding='UTF-8')as f_log:
            pass;    
    pass;

def run_one_model_experiment_v5(num_features_select_from_all_min:int=5,num_features_select_from_all_max:int=50,selected_indexes:list[int]=None,problem_type:str='regression',task_output:str='mono_output',score_type:str='mean_squared_error',fbeta_score_beta:float=1.0,d2_pinball_score_alpha:float=0.5,d2_tweedie_score_power:float=0.0,mean_pinball_loss_alpha:float=0.5,mean_tweedie_deviance_power:float=0.0,model_type:str=None,model_hyperparams:dict=None,num_folds:int=10,score_valid_min_threshold:float=None,score_valid_max_threshold:float=None,non_negative_y_guarantee:bool=False)->str:
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
    r2_score,root_mean_squared_error,root_mean_squared_log_error\n\n\n

    Сначала из матриц признаков opened_data_all_features и closed_data_all_features выбираются
    некоторым способом num_features_select_from_all столбцов (признаков), из которых составляются
    матрицы opened_data и closed_data соответственно\n
    Если num_features_select_from_all=0, то используются все признаки.
    Значение num_features_select_from_all определяется через num_features_select_from_all_min
    и num_features_select_from_all_max.
    Если в качестве параметра selected_indexes передаётся список, то отбираются именно эти признаки, а не случайные.
    """
    print(f'Функция run_one_model_experiment_v5 вызвана с параметрами: {locals()}');
    error_str:str='MODEL_WITH_ERROR';
    # 1. Загрузка ВСЕХ данных (открытых и закрытых) и отбор num_features_select_from_all признаков из всех
    #Загрузка выполняется отдельно, так как:
    #1) Если эксперимент повторяется много раз, загружать данные каждый раз неэффективно по времени
    #2) Данные могут быть представлены в разных форматах (csv,json,npy,...)
    #Отбор num_features_select_from_all признаков:
    n_samples_opened:int=opened_data_all_features.shape[0];
    n_samples_closed:int=closed_data_all_features.shape[0];
    n_features_all:int=opened_data_all_features.shape[1];#Равно opened_data_all_features.shape[1]    
    if selected_indexes is None:#Если индексы не заданы, то они выбираются случайным образом
        if num_features_select_from_all_min==num_features_select_from_all_max==0:num_features_select_from_all=0;
        else:num_features_select_from_all:int=random.randint(a=num_features_select_from_all_min,b=num_features_select_from_all_max);
        n_features_selected:int=num_features_select_from_all;
        if num_features_select_from_all==0:#Если num_features_select_from_all=0 и индексы не заданы, то используются все признаки
            all_indexes:list[int]=[i for i in range(n_features_all)];#Все индексы 0..n_features_all-1
            selected_indexes:list[int]=[i for i in range(n_features_all)];#Все индексы 0..n_features_all-1
        else:#Если num_features_select_from_all!=0 и индексы не заданы, то выполняется отбор случайного множества признаков из всех        
            all_indexes:list[int]=[i for i in range(n_features_all)];#Все индексы 0..n_features_all-1
            for i in range(10):random.shuffle(x=all_indexes);#Shuffle list x in place, and return None.
            selected_indexes:list[int]=[all_indexes[i] for i in range(num_features_select_from_all)];#Индексы выбранных признаков
            selected_indexes.sort();#Эта сортировка по идее ни на что не влияет, но так просто удобнее для человека
            #Использование списка selected_indexes и его сохранение в лог делает эсперименты воспроизводимыми
    else:#Если индексы заданы, то выбираются столбцы именно с этими индексами (ничего случайным образом не выбирается, ВНЕ зависимости
        #от num_features_select_from_all_min и num_features_select_from_all_max)
        n_features_selected:int=len(selected_indexes);
    #Для создания и заполнения массивов opened_data и closed_data используется один и тот же код вне зависимости от переданных
    #значений selected_indexes, num_features_select_from_all_min и num_features_select_from_all_max
    opened_data:np.ndarray=np.zeros((n_samples_opened,n_features_selected));#Создание новых массивов
    closed_data:np.ndarray=np.zeros((n_samples_closed,n_features_selected));
    for col_idx,feature_idx in enumerate(selected_indexes):#Копирование столбцов с выбранными индексами из исходных массивов
        opened_data[:,col_idx]=opened_data_all_features[:,feature_idx];
        closed_data[:,col_idx]=closed_data_all_features[:,feature_idx];




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
            if(non_negative_y_guarantee==False)and('PoissonRegressor'in model_types):model_types.remove('PoissonRegressor');#Some value(s) of y are negative which is not allowed for Poisson regression.
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
            elif model_type=='RandomForestRegressor':model_hyperparams={'n_estimators':random.randint(a=20,b=200),'criterion':random.choice(seq=['squared_error','absolute_error','friedman_mse','poisson']),'max_depth':random.randint(a=2,b=20),'min_samples_split':random.uniform(a=0.0,b=1.0),'min_samples_leaf':random.uniform(a=0.0,b=1.0),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_features':random.choice(seq=['sqrt','log2','sqrt','log2','sqrt','log2','sqrt','log2','sqrt','log2',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),'min_impurity_decrease':random.uniform(a=0.0,b=0.2),'bootstrap':random.choice(seq=[True,False]),'warm_start':random.choice(seq=[True,False]),'ccp_alpha':random.uniform(a=0.0,b=1.0),'max_samples':random.uniform(a=0.0,b=1.0),'n_jobs':-1,'random_state':hyperparam_random_state};
            
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
    with open(file=filename,mode='wb')as f:pickle.dump(obj={'n_features_selected':n_features_selected,'selected_indexes':selected_indexes,'model':model_production,'scaler':scaler_production},file=f);
    #print(f"Финальная модель сохранена в файл: {filename}")

    # 9. Логирование
    log_record_txt:str=f"""
--- Model ID: {model_id} ---
n_features_all: {n_features_all}
n_features_selected: {n_features_selected}
selected_indexes: {selected_indexes}
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
    log_record_csv:str=f'{model_id},{n_features_all},{n_features_selected},{model_type},{score_type},{score_valid_mean},{score_test}\n';
    print(log_record_txt);
    with open(file='log_models.txt',mode='at',encoding='UTF-8')as log_file:log_file.write(log_record_txt);
    with open(file='log_models.csv',mode='at',encoding='UTF-8')as log_file:log_file.write(log_record_csv);
    return model_id;

def float_list_to_comma_separated_str(float_list:list[float],digits:int=2):
    float_list_buf=list(np.round(np.array(float_list),digits));
    return ','.join([str(x)for x in float_list_buf]);

def create_predictions_files(model_ids:list[str],digits_round_min:int=1,digits_round_max:int=18)->None:
    '''Функция вычисляет предсказания и сохраняет их в файлах tsv и json, позволяя выбирать количество цифр'''
    targets_dict:dict[str:float]={};
    for sample_id in closed_ids:
        targets_dict[sample_id]=0.0;
    print(f'targets_dict: {targets_dict}');
    print(f'len(targets_dict): {len(targets_dict)}');
    model_ids_str:str=' '.join(model_ids);
    results_file_id:str=''.join(random.choices(population=string.ascii_uppercase+string.digits,k=16));
    #=======================Далее код, изначально перемещённый из основного цикла программы (затем доработанный):
    n_models:int=len(model_ids);
    print(f'Предсказания выполняются усреднением результатов для n_models={n_models} моделей со значениями id: {model_ids}');
    buf_s:str='';
    n_samples_opened:int=opened_ids.shape[0];
    n_samples_closed:int=closed_ids.shape[0];
    targets_list:list[float]=[0.0 for i in range(n_samples_closed)];#Изначально предсказания для всех образцов инициализируем нулями
    #так как 0+a=a для любого действительного числа a, все предсказания - это числа с плавающей точкой
    #Суммирование предсказаний моделей:
    for model_id in model_ids:#Перебираем модели
        pkl_file_name:str=f'model_{model_id}.pkl';
        with open(file=pkl_file_name,mode='rb')as f:model_dict:dict=pickle.load(file=f);
        print(f'model_id: {model_id}, model_dict: {model_dict}');
        n_features_selected:int=model_dict['n_features_selected'];
        selected_indexes:list[int]=model_dict['selected_indexes'];
        model=model_dict['model'];
        scaler=model_dict['scaler'];
        print(f'model_id: {model_id}');
        print(f'model.__dict__: {model.__dict__}');
        print(f'scaler.__dict__: {scaler.__dict__}');
        num_processed:int=0;
        opened_data:np.ndarray=np.zeros((n_samples_opened,n_features_selected));#Создание новых массивов
        closed_data:np.ndarray=np.zeros((n_samples_closed,n_features_selected));
        #Использование списка selected_indexes и его сохранение в лог делает эсперименты воспроизводимыми
        for col_idx,feature_idx in enumerate(selected_indexes):#Копирование столбцов с выбранными индексами из исходных массивов
            opened_data[:,col_idx]=opened_data_all_features[:,feature_idx];
            closed_data[:,col_idx]=closed_data_all_features[:,feature_idx];
        for sample_num in range(n_samples_closed):#Перебираем образцы закрытых данных
            sample_id:str=closed_ids[sample_num];
            features:np.ndarray=closed_data[sample_num].reshape(1, -1);#
            features_scaled=scaler.transform(features);
            target_predicted:float=model.predict(features_scaled)[0];
            num_processed=num_processed+1;
            #targets_list.append(target_predicted);
            targets_list[sample_num]=targets_list[sample_num]+target_predicted;
            print(f'sample_num: {sample_num:5d}, sample_id: {sample_id}, target_predicted: {target_predicted}, type(target_predicted): {type(target_predicted)}');
    #Приведение списка targets_list из list[numpy.float64] в list[float]
    for sample_num in range(n_samples_closed):targets_list[sample_num]=float(targets_list[sample_num]);
    #Усреднение предсказаний моделей:
    for sample_num in range(n_samples_closed):
        targets_list[sample_num]=targets_list[sample_num]/n_models;#Деление суммы предсказаний моделей на количество моделей
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
    model_ids_num:int=len(model_ids);
    results_log_record:str=f"""
results_file_id: {results_file_id}
digits_round_min: {digits_round_min}, digits_round_max: {digits_round_max}
model_ids_num: {model_ids_num}
model_ids_str: {model_ids_str}
================""";
    with open(file='log_results.txt',mode='at',encoding='UTF-8')as results_lof_file:print(results_log_record,file=results_lof_file);
    print(f'Модели с id {model_ids_str} применены, их усреднённые результаты в файлах {tsv_filename} и {json_filename}');
    pass;

def create_coefs_and_bias_files(model_ids:list[str],digits_round_min:int=2,digits_round_max:int=18)->None:
    """Функция принимает список id моделей с одинаковыми значениями n_features_selected, selected_indexes, scaler (mean,var,
    scale) и создаёт txt файл, в который записывает усреднённые значения coef: array([...]) (массив из n_features_selected чисел
    типа float) и intercept."""
    models_num:int=len(model_ids);#Количество моделей, по которым производится усреднение их coef и intercept
    #Считываем первую модель для инициализации некоторых значений:
    print(f'ИНИЦИАЛИЗАЦИЯ НЕКОТОРЫХ ЗНАЧЕНИЙ ПЕРЕД СУММИРОВАНИЕМ coefs и bias:');
    model_id:str=model_ids[0];
    pkl_file_name:str=f'model_{model_id}.pkl';
    with open(file=pkl_file_name,mode='rb')as f:model_dict:dict=pickle.load(file=f);
    print(f'model_id: {model_id}, model_dict: {model_dict}');
    n_features_selected:int=model_dict['n_features_selected'];#n_features_selected одинаковое у всех моделей
    selected_indexes:list[int]=model_dict['selected_indexes'];#selected_indexes одинаковое у всех моделей
    print(f'n_features_selected: {n_features_selected}, selected_indexes: {selected_indexes}');
    model=model_dict['model'];
    scaler=model_dict['scaler'];
    print(f'model_id: {model_id}');
    print(f'model.__dict__: {model.__dict__}');
    print(f'scaler.__dict__: {scaler.__dict__}');
    coefs_total:list[float]=[0.0 for i in range(n_features_selected)];#Инициализация массива весовых коэффициентов
    bias_total:float=0.0;#Инициализация смещения (intercept или bias) [все эти значения инициализируются нулями, так как a+0.0=a]
    #Вообще можно тут сохранить заодно и scaler (который должен быть одинаковым для всех моделей):
    scaler_for_all_models=model_dict['scaler'];
    scaler_mean_list_float:list[float]=[float(num)for num in scaler_for_all_models.__dict__['mean_']];
    scaler_var_list_float:list[float]=[float(num)for num in scaler_for_all_models.__dict__['var_']];
    scaler_scale_list_float:list[float]=[float(num)for num in scaler_for_all_models.__dict__['scale_']];
    scaler_scale_sqr_list_float:list[float]=[num**2 for num in scaler_scale_list_float];#Для проверки того, что var - это дисперсия,
    #а scale - это квадратный корень из дисперсии (или 1.0, если дисперсия равна 0.0)
    #Считывание моделей в цикле для суммирования соответствующих весовых коэффициентов coefs и смещения bias
    num_models_added:int=0;
    num_models_have_coef_and_intercept_attributes:int=0;
    for model_id in model_ids:
        pkl_file_name:str=f'model_{model_id}.pkl';
        print(f'Добавление модели номер {num_models_added+1} из {len(model_ids)}, model_id: {model_id}, file_name: {pkl_file_name}');
        with open(file=pkl_file_name,mode='rb')as f:model_dict:dict=pickle.load(file=f);
        model=model_dict['model'];
        coefs_current:np.ndarray=np.zeros(shape=(n_features_selected),dtype=np.float64);#Инициализация массива текущих считанных из
        #pkl файла коэффициентов (на тот случай, если такого атрибута нет)
        bias_current:np.ndarray=np.zeros(shape=(1),dtype=np.float64);#Инициализация текущего значения считанного из
        #pkl файла bias (на тот случай, если такого атрибута нет)
        if (hasattr(model,'coef_'))and(hasattr(model,'intercept_')):#Считывание coef и intercept, если они оба есть у модели
            num_models_have_coef_and_intercept_attributes=num_models_have_coef_and_intercept_attributes+1;
            coefs_current=model.__dict__['coef_'];
            bias_current=model.__dict__['intercept_'];
            print(f'model_id: {model_id}, coefs_current: {coefs_current}, bias_current: {bias_current}');
            for i in range(n_features_selected):coefs_total[i]=coefs_total[i]+float(coefs_current[i]);
            if type(bias_current)==np.ndarray:bias_total=bias_total+float(bias_current[0]);
            elif type(bias_current)==np.float64:bias_total=bias_total+float(bias_current);
            elif type(bias_current)==float:bias_total=bias_total+bias_current;
            else:print(f'bias_current: {bias_current}, type(bias_current): {type(bias_current)}, значение bias_current НЕ ДОБАВЛЕНО, так как его тип не предусмотрен в коле функции, НЕОБХОДИМО предусмотреть обработку случая с этим типом');
        else:
            print(f"hasattr(model,'coef_'): {hasattr(model,'coef_')}, hasattr(model,'intercept_'): {hasattr(model,'intercept_')}, эта модель НЕ добавлена, так как у неё нет хотя бы одного из нужных атрибутов");
        num_models_added=num_models_added+1;
    print(f'Все модели обработаны, из имеющих атрибуты coef_ и intercept_ (таких {num_models_have_coef_and_intercept_attributes} из {models_num} или {(100*num_models_have_coef_and_intercept_attributes/models_num):.4f}%) добавлены коэффициенты и смещение');
    for i in range(n_features_selected):coefs_total[i]=coefs_total[i]/num_models_have_coef_and_intercept_attributes;
    bias_total=bias_total/num_models_have_coef_and_intercept_attributes;
    print(f'После усреднения по {num_models_have_coef_and_intercept_attributes} моделям (до учёта scaler):');
    print(f'coefs: {coefs_total}, bias: {bias_total}');
    str_list_to_txt:list[str]=[];
    str_list_to_txt.append(f'n_features_selected: {n_features_selected}, models_num: {models_num}, model_ids: {model_ids}\n');
    str_list_to_txt.append(f'{num_models_have_coef_and_intercept_attributes} из {models_num} моделей (или {(100*num_models_have_coef_and_intercept_attributes/models_num):.4f}%) имеют атрибуты coef_ и intercept_, по этим моделям выполняется усреднение\n');
    str_list_to_txt.append(f'После усреднения по {num_models_have_coef_and_intercept_attributes} моделям (до учёта scaler):\n');
    str_list_to_txt.append(f'coefs: {coefs_total}, bias: {bias_total}\n');
    str_list_to_txt.append(f'scaler_mean_list_float: {scaler_mean_list_float}\n');
    str_list_to_txt.append(f'scaler_var_list_float: {scaler_var_list_float}\n');
    str_list_to_txt.append(f'scaler_scale_list_float: {scaler_scale_list_float}\n');
    str_list_to_txt.append(f'scaler_scale_sqr_list_float: {scaler_scale_sqr_list_float}\n');
    
    #Вот что сказано про решение задачи 2:
    """Сдача второй части соревнования
    Для сдачи вам достаточно отправить функцию my_transformation и параметры вашей модели в контест в задачу №2. Пример посылки
    доступен ниже. Имортирование numpy также необходимо.
    # __________example_submission_start__________
    import numpy as np
    def my_transformation(feature_matrix:np.ndarray)->np.ndarray:
        new_feature_matrix = np.zeros((feature_matrix.shape[0], feature_matrix.shape[1]+1))
        new_feature_matrix[:, :feature_matrix.shape[1]] = feature_matrix
        new_feature_matrix[:, -1] = feature_matrix[:, 0
        ] * feature_matrix[:, 1]
        return new_feature_matrix

    w_submission = [-0.0027, -0.2637, 0.0, -0.1134, -0.0165, -0.9329, 0.0, 0.1293]
    b_submission = 1.1312
    # __________example_submission_end__________
    """
    #То есть коэффициенты записываются в виде списка w_submission:list[float], с точностью 4 цифры после точки
    #смещение (bias) записывается в виде числа b_submission:float, с точностью 4 цифры после точки

    #На этот момент есть список весовых коэффициентов coefs_total и смещение bias_total
    #Теперь нужно изменить эти значения (coefs_total и bias_total) так, чтобы учесть содержимое списков:
    #scaler_mean_list_float, scaler_var_list_float, scaler_scale_list_float
    #StandardScaler сначала вычитает из признака среднее (mean), затем делит на корень из дисперсии
    #var - это дисперсия, scale - это корень из дисперсии
    #НО если var=0.0, то scale=1.0 (то есть у какого-то признака значения для всех образцов, на которых обучился scaler, одинаковые)
    #Например (данные из файла "log_results.txt", а не выдуманные):
    #scaler_mean_list_float:      [0.12757897540767849, 0.3330816148209652, 0.3233406235457518, 0.06354032805974918, 0.8575786907593768, 1.0327984875202796, -1.8122844359820223, 0.8914758767871305, 0.0, 0.362167046340388, -1.245160663783746, -0.5170342419190717, -0.0006947738260783553, -0.13434136275458952]
    #scaler_scale_list_float:     [0.13279250391936917, 1.073656562593154e-11, 1.4126706005062035e-12, 0.08206730267207896, 5.479094239864417e-13, 0.2642597116483548, 1.012380145794119, 0.18019845350340546, 1.0, 0.4006329073243955, 0.03210874262474023, 9.386581322983104e-13, 0.262727234158622, 0.2741375772195151]
    #scaler_var_list_float:       [0.017633849097175677, 1.152738414399347e-22, 1.9956382255345572e-24, 0.0067350421678706186, 3.002047368931544e-25, 0.06983319520047164, 1.0249135595981218, 0.032471482645018986, 0.0, 0.16050672643119768, 0.00103097135294181, 8.810790893297524e-25, 0.06902559956863938, 0.0751514112437856]
    #scaler_scale_sqr_list_float: [0.017633849097175677, 1.152738414399347e-22, 1.9956382255345576e-24, 0.006735042167870618, 3.0020473689315435e-25, 0.06983319520047164, 1.0249135595981218, 0.03247148264501898, 1.0, 0.16050672643119768, 0.00103097135294181, 8.810790893297522e-25, 0.06902559956863938, 0.0751514112437856]
    #Вообще если у какого-то признака дисперсия точно равна нулю, значит этот признак явяется постоянным (значения этого признака для
    #всех образцов, на которых обучался scaler, равны между собой), этот признак не несёт никакой информации, значит скорее всего
    #этот признак не имеет смысла и лучше убрать этот признак из модели

    #Непонятно тогда, как так получилось, что этот не несущий никакой информации признак попал в 14 наиболее часто используемых
    #признаков в 1276 моделях, показавших хорошие результаты в первой части задания

    #Прямое преобразование признаков:
    #1) feature_value=feature_value-mean (вычитание среднего)
    #2) feature_value=feature_value/scale (деление на корень из дисперсии или на 1.0, если дисперсия равна 0.0)
    
    #Обратное преобразование признаков (порядок, обратный порядку прямого преобразования):
    #1) feature_value=feature_value*scale
    #2) feature_value=feature_value+mean

    #Но модель по условию задачи должна быть представлена не ответами на закрытых данных, а списком коэффициентов (coefs_total и
    #bias_total). То есть нужно так преобразовать coefs_total и bias_total, чтобы эти преобразования были эквивалентны обратному
    #преобразованию признаков, применённому ПЕРЕД умножением вектора признаков на вектор coefs_total и добавлением bias_total.
    #1) feature_value=feature_value*scale эквивалентно:
    #1) c=coefs_total*scale (каждый коэффициент умножается на свой scale)
    #+ тут скорее всего нужно как-то тоже учесть bias
    #2) feature_value=feature_value+mean эквивалентно:
    #2) bias_total=bias_total+mean НО значение mean у каждого признака своё, а значение bias_total одно у всей модели!!!

    # ========== ПРЕОБРАЗОВАНИЕ КОЭФФИЦИЕНТОВ С УЧЕТОМ SCALER ==========
    # StandardScaler выполняет: x_scaled = (x - mean) / scale
    # Модель: y_pred = coefs @ x_scaled + bias       #в NumPy оператор @ означает умножение матриц
    # Подставляем x_scaled: y_pred = coefs @ ((x - mean) / scale) + bias
    # Раскрываем: y_pred = (coefs / scale) @ x - (coefs / scale) @ mean + bias
    # 
    # Таким образом, новые коэффициенты и смещение:
    # new_coefs = coefs / scale
    # new_bias = bias - (coefs / scale) @ mean
    coefs_after_scaler:list[float]=[0.0]*n_features_selected;
    bias_after_scaler:float=bias_total;
    for i in range(n_features_selected):
        #Каждый коэффициент делится на соответствующий scale
        coefs_after_scaler[i]=coefs_total[i]/scaler_scale_list_float[i] if scaler_scale_list_float[i]!=0.0 else 0.0;
        #Вычитаем вклад mean из bias
        bias_after_scaler=bias_after_scaler-coefs_after_scaler[i]*scaler_mean_list_float[i];
    str_list_to_txt.append(f'После усреднения по {num_models_have_coef_and_intercept_attributes} моделям (после учёта scaler):\n');
    print(f'coefs: {coefs_after_scaler}, bias: {bias_after_scaler}');
    str_list_to_txt.append(f'coefs: {coefs_after_scaler}, bias: {bias_after_scaler}\n');
    #Результат этого кода (из лога):
    """
    n_features_selected: 14, models_num: 5, model_ids: ['0UVA9BKRDG2E7EX6', '0271ZHM0Z5R3HXLS', '04HVI5VS4FKMIVBG', '05DZJE2OQWUXM93R', '06SJSYUV7IEKGEXX']
    3 из 5 моделей (или 60.0000%) имеют атрибуты coef_ и intercept_, по этим моделям выполняется усреднение
    После усреднения по 3 моделям (до учёта scaler):
    coefs: [-1.2313695017445994, 0.4978657583672675, -346.63764278718253, -0.3225509807834996, 14.809532910787548, 9.398633209534827, 0.003607184926051259, -0.4164114424501501, -4.95391825947184e-11, 0.2248962733541092, -1.9012442727975356, -360.9458753751258, -12.266215347254906, -3.487854693156537], bias: 2.4712619909030535
    scaler_mean_list_float: [0.12757897540767849, 0.3330816148209652, 0.3233406235457518, 0.06354032805974918, 0.8575786907593768, 1.0327984875202796, -1.8122844359820223, 0.8914758767871305, 0.0, 0.362167046340388, -1.245160663783746, -0.5170342419190717, -0.0006947738260783553, -0.13434136275458952]
    scaler_var_list_float: [0.017633849097175677, 1.152738414399347e-22, 1.9956382255345572e-24, 0.0067350421678706186, 3.002047368931544e-25, 0.06983319520047164, 1.0249135595981218, 0.032471482645018986, 0.0, 0.16050672643119768, 0.00103097135294181, 8.810790893297524e-25, 0.06902559956863938, 0.0751514112437856]
    scaler_scale_list_float: [0.13279250391936917, 1.073656562593154e-11, 1.4126706005062035e-12, 0.08206730267207896, 5.479094239864417e-13, 0.2642597116483548, 1.012380145794119, 0.18019845350340546, 1.0, 0.4006329073243955, 0.03210874262474023, 9.386581322983104e-13, 0.262727234158622, 0.2741375772195151]
    scaler_scale_sqr_list_float: [0.017633849097175677, 1.152738414399347e-22, 1.9956382255345576e-24, 0.006735042167870618, 3.0020473689315435e-25, 0.06983319520047164, 1.0249135595981218, 0.03247148264501898, 1.0, 0.16050672643119768, 0.00103097135294181, 8.810790893297522e-25, 0.06902559956863938, 0.0751514112437856]
    После усреднения по 3 моделям (после учёта scaler):
    coefs: [-9.272884126744682, 46371044122.7868, -245377544250564.53, -3.9303226776239386, 27029162599608.83, 35.56589519798388, 0.0035630735559533857, -2.3108491463400966, -4.95391825947184e-11, 0.5613524731557036, -59.212666625338386, -384533903191514.0, -46.68802374651864, -12.723008383355065], bias: -142671746183561.94
    ================================
    Проверка:
    Пусть для некоторого образца значения всех признаков равны по 1.0. Тогда полученные после учёта scaler коэффициенты дадут резуьтат:
    y=(1.0)*(-9.272884126744682)+(1.0)*(46371044122.7868)+(1.0)*(-245377544250564.53)+(1.0)*(-3.9303226776239386)+(1.0)*(27029162599608.83)+(1.0)*(35.56589519798388)+(1.0)*(0.0035630735559533857)+(1.0)*(-2.3108491463400966)+(1.0)*(-4.95391825947184e-11)+(1.0)*(0.5613524731557036)+(1.0)*(-59.212666625338386)+(1.0)*(-384533903191514.0)+(1.0)*(-46.68802374651864)+(1.0)*(-12.723008383355065)+(-142671746183561.94)=-745507659982006.8601439612748103968947184
    Если вместо этого применить ко всем признакам scaler, то они будут иметь значения:
    (1.0-(0.12757897540767849))/(0.13279250391936917)=6.56980626799574803053416027287831462350619577965366234721
    (1.0-(0.3330816148209652))/(1.073656562593154e-11)=62116547172.985844963479815360670327276923915167561255982618742444
    (1.0-(0.3233406235457518))/(1.4126706005062035e-12)=478993033628.4907818578798670445924964549362195410407021931063072
    (1.0-(0.06354032805974918))/(0.08206730267207896)=11.41087426355556701813846172546529563686008091610638244678
    (1.0-(0.8575786907593768))/(5.479094239864417e-13)=259935863494.3783854647037983118284410246815969201963215283610412698
    (1.0-(1.0327984875202796))/(0.2642597116483548)=-0.1241145966431837409991261749365182219330767205835036633
    (1.0-(-1.8122844359820223))/(1.012380145794119)=2.7778937068901564909474748910304658207315765886602
    (1.0-(0.8914758767871305))/(0.18019845350340546)=0.6022478056995010181239018084182380474619557545697261389
    (1.0-(0.0))/(1.0)=1.000000
    (1.0-(0.362167046340388))/(0.4006329073243955)=1.59206331282006703859220915915017688178295048375938546243
    (1.0-(-1.245160663783746))/(0.03210874262474023)=69.923655685408829304395679417807873927681213627411352078
    (1.0-(-0.5170342419190717))/(9.386581322983104e-13)=1616173332675.021651717157041651873949824804608711558690521642913127
    (1.0-(-0.0006947738260783553))/(0.262727234158622)=3.8088734006993246795810735048331188936292802247319983287
    (1.0-(-0.13434136275458952))/(0.2741375772195151)=4.137854336716005961818041018473423451997989329249510995
	
	С коэффициентами, определёнными моделью до scaler, это даст результат:	y=(6.56980626799574803053416027287831462350619577965366234721)*(-1.2313695017445994)+(62116547172.985844963479815360670327276923915167561255982618742444)*(0.4978657583672675)+(478993033628.4907818578798670445924964549362195410407021931063072)*(-346.63764278718253)+(11.41087426355556701813846172546529563686008091610638244678)*(-0.3225509807834996)+(259935863494.3783854647037983118284410246815969201963215283610412698)*(14.809532910787548)+(-0.1241145966431837409991261749365182219330767205835036633)*(9.398633209534827)+(2.7778937068901564909474748910304658207315765886602)*(0.003607184926051259)+(0.6022478056995010181239018084182380474619557545697261389)*(-0.4164114424501501)+(1.000000)*(-4.95391825947184e-11)+(1.59206331282006703859220915915017688178295048375938546243)*(0.2248962733541092)+(69.923655685408829304395679417807873927681213627411352078)*(-1.9012442727975356)+(1616173332675.021651717157041651873949824804608711558690521642913127)*(-360.9458753751258)+(3.8088734006993246795810735048331188936292802247319983287)*(-12.266215347254906)+(4.137854336716005961818041018473423451997989329249510995)*(-3.487854693156537)+(2.4712619909030535)=-745507659982006.8238272765989945936177245244481607361770875403608003845251195268

	Результат, полученный из исходных коэффициентов и смещения (с применением scaler к признакам): -745507659982006.8238272765989945936177245244481607361770875403608003845251195268
	Результат, полученный из коэффициентов, в которых уже учтён scaler: -745507659982006.8601439612748103968947184
	Погрешность (абсолютная): abs((-745507659982006.8238272765989945936177245244481607361770875403608003845251195268)-(-745507659982006.8601439612748103968947184))=0.0363166846758158032769938755518392638229124596391996154748804732
	Погрешность (относительная): 0.0363166846758158032769938755518392638229124596391996154748804732/abs(-745507659982006.8601439612748103968947184)=0.0000000000000000487140329003357546774519909765492698989474250221=4.87140329003357546774519909765492698989474250221e-17
	
	Вывод: это преобразование коэффициентов ПРАВИЛЬНОЕ, его применение к коэффициентам (coef и bias) ЭКВИВАЛЕНТНО применению к признакам scaler с теми параметрами mean и scale, которые заданы в списках scaler_mean_list_float и scaler_scale_list_float соответственно.




    
    """









        
        





    
    





    str_list_to_txt.append(f'================================\n');
    with open(file='log_results.txt',mode='at',encoding='UTF-8')as f_log:f_log.writelines(str_list_to_txt);
    pass;

#Действия, выполняемые перед каждым запуском:
opened_data_all_features,opened_target,opened_ids,closed_data_all_features,closed_target,closed_ids=load_data_from_npy();
create_log_files();

#Основной цикл программы:
command_num:int=0;
while command_num>-1:
    print(f'=====================================');
    print(f'1 => выполнить кросс-валидацию n раз v5');
    print(f'2 => создать json файл с предсказанием модели или средним предсказанием нескольких моделей из списка их id');
    print(f'3 => анализ csv лога с результатами моделей');
    print(f'4 => анализ txt лога с результатами моделей');
    print(f'5 => вывод информации о содержимом одного pkl файла');
    print(f'6 => создать txt файл со значениями coef и bias для модели или усреднёнными значениями нескольких моделей из списка их id');

    print(f'-1 => выйти из программы');
    print(f'=====================================');

    input_str:str=input('Введите номер команды: ');
    print(f'Введено: {input_str}');
    command_num=int(input_str);
    if command_num==1:#1 => выполнить кросс-валидацию n раз v5
        num_of_experiments:int=int(input('Введите количество экспериментов: '));
        num_features_select_from_all_min:int=0;num_features_select_from_all_max:int=0;
        selected_indexes_str:str=input('Введите список номеров признаков для отбора через ПРОБЕЛ (список для отбора одинакового набора признаков во всех экспериментах или просто Enter для случайного задания списка в каждом эксперименте), пример списка (без квадратных скабок): [218 307 56 266 63 67 77 336 84 105 376 59 73 257]: ');
        if len(selected_indexes_str)==0:#Если список индексов не задан, то они могут выбираться случайно (если их количество равно 0)
            selected_indexes:list[int]=None;
            num_features_select_from_all_min:int=int(input('Введите минимальное  количество случайно отбираемых признаков в каждом эксперименте (или 0 для использования всех признаков): '));
            num_features_select_from_all_max:int=int(input('Введите максимальное количество случайно отбираемых признаков в каждом эксперименте (или 0 для использования всех признаков): '));
        else:selected_indexes:list[int]=[int(num)for num in selected_indexes_str.split(sep=' ')];
        for i in range(num_of_experiments):
            try:
                print(f'Эксперимент {i+1}/{num_of_experiments}... ',end='');
                model_id:str=run_one_model_experiment_v5(num_features_select_from_all_min=num_features_select_from_all_min,num_features_select_from_all_max=num_features_select_from_all_max,selected_indexes=selected_indexes,problem_type='regression',task_output='mono_output',score_type='mean_squared_error',
                model_type=None,model_hyperparams=None,num_folds=10,score_valid_min_threshold=None,score_valid_max_threshold=0.10);
            except Exception as ex:
                print(f'Возникло исключение, type(ex): {type(ex)}, ex: {ex}');
    elif command_num==2:#2 => создать json файл с предсказанием модели или средним предсказанием нескольких моделей из списка их id
        model_ids_str:str=input('Введите id модели или нескольких моделей через запятую или пробел (например, [08JZRAWXBE5N43MX] или [08JZRAWXBE5N43MX,2352C29OXLDYGPAL,J0KZOWU71FHE3TCR,EENT8VMHI4CK4D24]) (БЕЗ КВАДРАТНЫХ СКОБОК): ');
        digits_round_min,digits_round_max=[int(num_s)for num_s in input('Введите минимальное и максимальное количество цифр округления (например, 2 18): ').split(sep=' ')]
        if ','in model_ids_str:model_ids_list:list[str]=model_ids_str.split(sep=',');
        elif ' 'in model_ids_str:model_ids_list:list[str]=model_ids_str.split(sep=' ');
        elif '/'in model_ids_str:model_ids_list:list[str]=model_ids_str.split(sep='/');
        elif '|'in model_ids_str:model_ids_list:list[str]=model_ids_str.split(sep='|');
        create_predictions_files(model_ids=model_ids_list,digits_round_min=digits_round_min,digits_round_max=digits_round_max);
    elif command_num==3:#анализ csv лога с результатами моделей
        score_valid_mean_threshold_min:float=0.0;
        score_valid_mean_threshold_max:float=0.090;
        score_test_threshold_min:float=0.0;
        score_test_threshold_max:float=0.115;
        n_features_selected_threshold_min:float=None;
        n_features_selected_threshold_max:float=None;       
        analize_log_models_csv(log_models_csv_file_name='log_models.csv',score_valid_mean_threshold_min=score_valid_mean_threshold_min,score_valid_mean_threshold_max=score_valid_mean_threshold_max,score_test_threshold_min=score_test_threshold_min,score_test_threshold_max=score_test_threshold_max,n_features_selected_threshold_min=n_features_selected_threshold_min,n_features_selected_threshold_max=n_features_selected_threshold_max);
        pass;
    elif command_num==4:#анализ txt лога с результатами моделей
        analize_log_models_txt(log_models_txt_file_name='log_models.txt');

        pass;
    elif command_num==5:
        pkl_file_name:str=input('Введите название pkl файла (например, model_RM3W9PGWI65QRNXI.pkl): ');
        analize_one_pkl_file(pkl_file_name=pkl_file_name);

        pass;
    elif command_num==6:#6 => создать txt файл со значениями coef и bias для модели или усреднёнными значениями нескольких моделей из списка их id
        model_ids_str:str=input('Введите id модели или нескольких моделей через запятую или пробел (например, [93P121PG8FACD4L2] или [93P121PG8FACD4L2 0UVA9BKRDG2E7EX6 0271ZHM0Z5R3HXLS 04HVI5VS4FKMIVBG 05DZJE2OQWUXM93R 06SJSYUV7IEKGEXX]) (БЕЗ КВАДРАТНЫХ СКОБОК): ');
        digits_round_min,digits_round_max=[int(num_s)for num_s in input('Введите минимальное и максимальное количество цифр округления (например, 2 18): ').split(sep=' ')]
        if ','in model_ids_str:model_ids_list:list[str]=model_ids_str.split(sep=',');
        elif ' 'in model_ids_str:model_ids_list:list[str]=model_ids_str.split(sep=' ');
        elif '/'in model_ids_str:model_ids_list:list[str]=model_ids_str.split(sep='/');
        elif '|'in model_ids_str:model_ids_list:list[str]=model_ids_str.split(sep='|');
        create_coefs_and_bias_files(model_ids=model_ids_list,digits_round_min=digits_round_min,digits_round_max=digits_round_max);

print(f'Работа программы завершена');



