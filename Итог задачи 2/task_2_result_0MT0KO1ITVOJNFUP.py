# ___________submission_start__________
import numpy as np;
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
def my_transformation(feature_matrix:np.ndarray)->np.ndarray:
    n_samples:int=feature_matrix.shape[0];#Количество строк таблицы, остаётся постоянным при добавлении новых признаков
    buf_feature_matrix:np.ndarray=data_transformation_pairwise_multiplications(feature_matrix=feature_matrix);
    buf_feature_matrix:np.ndarray=data_transformation_add_functions(feature_matrix=buf_feature_matrix);
    new_feature_matrix:np.ndarray=np.zeros(shape=(n_samples,14),dtype=np.float64);#В итоговой матрице 14 признаков по условию
    selected_indexes:list[int]=[218, 307, 56, 266, 63, 67, 77, 336, 105, 376, 59, 73, 257, 42];
    for new_col_idx,original_idx in enumerate(selected_indexes):
        new_feature_matrix[:,new_col_idx]=buf_feature_matrix[:,original_idx];    
    return new_feature_matrix;
#После усреднения по 748 моделям (после учёта scaler):
#coefs: [-7.442166495316138, 60074472564.38613, -72697493597795.88, -4.8062872882106715, 50095398007561.02, -11.56272211926139, 0.006217737790354939, 0.41255834406342773, 0.49396605123514326, -125.63055647195807, -137960723496339.95, -60.37429485244821, 358.7009577529351, 956.2311127947322], bias: -90805120716765.53
w_submission:list[float]=[-7.4422,6.0074e10,-7.2697e13,-4.8063,5.0095e13,-11.5627,6.2177e-3,0.4126,0.4940,-125.6306,-1.3796e14,-60.3743,358.7010,956.2311];
b_submission:float=-9.0805e13;
# ___________submission_end__________