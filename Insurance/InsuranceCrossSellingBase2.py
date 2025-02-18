"""
@Project
@File    InsuranceCrossSellingBase2.py.py
@Author  zhouhan
@Date    2025/2/12 18:29
"""
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings

import gc




print("InsuranceCrossSellingBaseModel2")



warnings.filterwarnings('ignore')
SEED = 90
N_FOLDS = 5
# 目标标签
TARGET = 'Response'


def reduce_mem_usage(df):
	""" iterate through all the columns of a dataframe and modify the data type
		to reduce memory usage.
	"""
	start_mem = df.memory_usage().sum() / 1024 ** 2
	print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

	for col in df.columns:
		col_type = df[col].dtype

		if col_type != object:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)
		else:
			df[col] = df[col].astype('category')
	end_mem = df.memory_usage().sum() / 1024 ** 2
	print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
	print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

	return df


def import_data(file, **kwargs):
	"""create a dataframe and optimize its memory usage"""
	df = pd.read_csv(file, parse_dates=True, keep_date_col=True, **kwargs)
	df = reduce_mem_usage(df)
	return df


def print_fmt(display_text, data):
	print(f"---------------------{display_text} START---------------------------")
	print(data)
	print(f"---------------------{display_text} END-----------------------------\n")
	pass


train = import_data("./data/train.csv", index_col="id", engine="pyarrow")
test = import_data("./data/test.csv", index_col="id", engine="pyarrow")

train["Region_Code"] = train["Region_Code"].astype(np.int8)
test["Region_Code"] = test["Region_Code"].astype(np.int8)

train["Policy_Sales_Channel"] = train["Policy_Sales_Channel"].astype(np.int16)
test["Policy_Sales_Channel"] = test["Policy_Sales_Channel"].astype(np.int16)
# 初始特征
initial_features = test.columns.to_list()
print_fmt("初始特征", initial_features)
# 分类特征
categorical_features = [col for col in initial_features if pd.concat([train[col], test[col]]).nunique() < 10]
print_fmt("分类特征", categorical_features)
# 数值特征
numerical_features = list(set(initial_features) - set(categorical_features))
print_fmt("数值特征", numerical_features)
# 特征分布
train[categorical_features] = train[categorical_features].astype("category")
test[categorical_features] = test[categorical_features].astype("category")
# 数值类型特征统计
print_fmt("数值类型特征统计 train", train.describe().T)
print_fmt("数值类型特征统计 test", test.describe().T)


gender_map = {
	'Female': 0,
	'Male': 1
}

# 年龄分布map
vehicle_age_map = {
	'< 1 Year': 0,
	'1-2 Year': 1,
	'> 2 Years': 2
}

# 车辆损坏map
vehicle_damage_map = {
	'No': 0,
	'Yes': 1
}
train_copy = train.copy()
train_copy['Gender'] = train_copy['Gender'].map(gender_map)
train_copy['Driving_License'] = train_copy['Driving_License'].astype(int)
train_copy['Previously_Insured'] = train_copy['Previously_Insured'].astype(int)
train_copy['Vehicle_Age'] = train_copy['Vehicle_Age'].map(vehicle_age_map)
train_copy['Vehicle_Damage'] = train_copy['Vehicle_Damage'].map(vehicle_damage_map)

X_copy = train_copy.sample(frac=0.05)
y_copy = X_copy.pop(TARGET)

mi_scores = mutual_info_classif(X_copy, y_copy, discrete_features=X_copy.dtypes == int, n_neighbors=5, random_state=42)
mi_scores = pd.Series(mi_scores, index=initial_features)
mi_scores = mi_scores.sort_values(ascending=False)
print_fmt("互信息评分", mi_scores)

del train_copy, X_copy, y_copy
gc.collect()

# 0.859185
xgb_params = {
	'eval_metric': 'auc',
	'n_estimators': 5000,						# R1 2000 -> 5000 AUC 0.859185
	'eta': 0.02,								# R1 0.1  -> 0.02 AUC 0.859185
	'alpha': 0.1269124823585012,
	'subsample': 0.8345882521794742,
	'colsample_bytree': 0.44370196445757065,    # R5 0.442 -> 0.443 AUC 0.859267 -> 0.859267
	'max_depth': 8,								# R1 15 -> 8 	  AUC 0.859185
	'tree_method': 'hist',
	'min_child_weight': 8,
	'gamma': 1.308021832047589e-08,
	'max_bin': 50000,
	'n_jobs': -1,
	'device': 'cuda',
	'enable_categorical': True,
	'early_stopping_rounds': 200,			   # R1 50 -> 200	AUC 0.859185
	'random_state': 90,					       # R2 add param   AUC 0.859185 -> 0.859212
	'reg_alpha': 0.1,              			   # R3 add L1正则化 AUC 0.859212 -> 0.859267
	'reg_lambda': 1.0,             			   # R4 add L2正则化 AUC 无变化 该参数和L1 非关键参数
}

# 去除目标变量列，得到特征矩阵
X = train.drop(TARGET, axis=1)
# 提取目标变量列
y = train[TARGET]
# random_state 保证每次划分的结果一致，方便复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
xgb_clf = xgb.XGBClassifier(**xgb_params)
xgb_clf = xgb_clf.fit(
	X_train,
	y_train,
	eval_set=[(X_test, y_test)],
	verbose=500
)
y_test_pred = xgb_clf.predict(X_test, iteration_range=(0, xgb_clf.best_iteration + 1))
y_test_prob = xgb_clf.get_booster().predict(xgb.DMatrix(X_test, enable_categorical=True), iteration_range=(0, xgb_clf.best_iteration + 1))
print(f"AUC: {roc_auc_score(y_test, y_test_prob):.6f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred), end="\n\n")

ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred), ).plot()


# 获取基本的预测结果
test_pred_base = xgb_clf.get_booster().predict(xgb.DMatrix(test, enable_categorical=True), iteration_range=(0, xgb_clf.best_iteration + 1))

# 两次判断进行数据拟合
# Case 1 欠采样
rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X, y)
X_train_rus, X_test_rus, y_train_rus, y_test_rus = train_test_split(X_rus, y_rus, test_size=0.2, shuffle=True, random_state=42, stratify=y_rus)
xgb_clf_rus = xgb.XGBClassifier(**xgb_params)

xgb_clf_rus = xgb_clf_rus.fit(
	X_train_rus,
	y_train_rus,
	eval_set=[(X_test_rus, y_test_rus)],
	verbose=500
)
test_pred_rus = xgb_clf_rus.get_booster().predict(xgb.DMatrix(test, enable_categorical=True), iteration_range=(0, xgb_clf_rus.best_iteration + 1))

# Case 2 过采样
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(X, y)
X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(X_ros, y_ros, test_size=0.2, shuffle=True, random_state=42, stratify=y_ros)
xgb_clf_ros = xgb.XGBClassifier(**xgb_params)
xgb_clf_ros = xgb_clf_ros.fit(
	X_train_ros,
	y_train_ros,
	eval_set=[(X_test_ros, y_test_ros)],
	verbose=500
)

test_pred_ros = xgb_clf_ros.get_booster().predict(xgb.DMatrix(test, enable_categorical=True), iteration_range=(0, xgb_clf_ros.best_iteration + 1))
sub = pd.DataFrame({
	'id': test.index,
	'Response': test_pred_ros
})

# PS 训练AUC为0.85
print_fmt("result", sub)
