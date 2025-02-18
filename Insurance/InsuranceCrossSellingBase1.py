"""
@Project 
@File    InsuranceCrossSellingBase1.py
@Author  zhouhan
@Date    2025/2/12 09:30
"""

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import gc

matplotlib.use('TkAgg')  # 或 'Qt5Agg'

warnings.filterwarnings('ignore')
# MatLib 绘图大小
fig_size = (15, 20)
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


# 加载所有数据 pandas 中 read_csv 的 engine=“pyarrow” 这个参数。采用C++ 的数据结构，有些数据类型速度可以提升31倍
# https://datapythonista.me/blog/pandas-20-and-the-arrow-revolution-part-i
# reduce_mem_usage 根据数据的实际大小重新定义其类型，尽量采用小类型，以减少内存使用量。从上述结果可以看出，减少了70%+的大小
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

plt.figure(figsize=fig_size)

# 分类特征饼图
for i, col in enumerate(categorical_features):
	plt.subplot(3, 2, i + 1)
	train[col].value_counts().plot(kind='pie', autopct='%.2f%%', pctdistance=0.8, fontsize=12)
	plt.gca().add_artist(plt.Circle((0, 0), radius=0.6, fc='white'))
	plt.xlabel(' '.join(col.split('_')), weight='bold', size=20)
	plt.ylabel("")

plt.tight_layout()
plt.suptitle("Pie Chart of Categorical Features", size=28, y=1.02)
plt.show()

# 分类特征的 Histrogram
for i, col in enumerate(categorical_features):
	plt.subplot(2, 3, i + 1)
	sns.countplot(x=train[col], hue=train[TARGET])
	plt.xlabel(' '.join(col.split('_')))
	plt.ylabel("Frequency")

plt.tight_layout()
plt.suptitle("Histrogram of Categorical Features", size=28, y=1.03)
plt.show()

# 抽取样本 frac为抽样数
# 数值特征的直方图
train_sampled = train.sample(frac=0.95)
for i, col in enumerate(numerical_features):
	plt.subplot(2, 3, i + 1)
	sns.histplot(data=train_sampled, x=col, hue=TARGET)
	plt.xlabel(' '.join(col.split('_')))
	plt.ylabel("Frequency")
plt.tight_layout()
plt.suptitle("Histogram of Numerical Features (95% Data)", size=28, y=1.03)
plt.show()

# 数值特征的小提琴图 fig-size单位为英制单位 英寸
for i, col in enumerate(numerical_features):
	plt.subplot(3, 2, i + 1)
	sns.violinplot(data=train_sampled, x=col, hue=TARGET)
	plt.xlabel(' '.join(col.split('_')), weight="bold", size=20)
	plt.ylabel("")

plt.tight_layout()
plt.suptitle("Violin Plot of Numerical Features (95% Data)", size=25, y=1.02)
plt.show()

# 数据分类相关性 map编码，构建新的数据结果集
# 年龄map
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
# 查看分布
cor_mat = train_copy.corr(method="pearson")
mask = np.triu(np.ones_like(cor_mat))
plt.figure(figsize=(16, 12))
sns.heatmap(cor_mat, cmap='coolwarm', fmt='.2f', annot=True, mask=mask)
plt.show()

# 根据上述图标 进行特征的选择模块 进行互信评分`互信息评分` 了解每个特征对目标变量的描述程度,丢弃那些不能帮助我们显著理解目标变量的特征
X_copy = train_copy.sample(frac=0.05)
y_copy = X_copy.pop(TARGET)

mi_scores = mutual_info_classif(X_copy, y_copy, discrete_features=X_copy.dtypes == int, n_neighbors=5, random_state=42)
mi_scores = pd.Series(mi_scores, index=initial_features)
mi_scores = mi_scores.sort_values(ascending=False)
print_fmt("互信息评分", mi_scores)
# 信息评分图
mi_scores.plot(kind='barh', title='Mutual Info Score of Features', figsize=(15, 20), xlabel="Score", ylabel="Feature")
plt.show()

# 特征选择是非常重要的一项工作，它可以过滤掉一些无用或冗余的特征，提高模型的准确性和可解释性。其中，互信息法（mutual information）是一种常用的特征选择方法。
# 互信息指的是两个变量之间的相关性，它测量了一个随机变量中的信息量能够为另一个随机变量提供多少信息。在特征选择中，我们可以通过计算每个特征与目标变量之间的互信息来判断该特征是否有预测能力。
# 在Python中，使用sklearn库中的mutual_info_classif和mutual_info_regression函数来计算互信息
del train_copy, X_copy, y_copy
gc.collect()

"""
xgb_params 参数
	'eval_metric': 'auc',								# 分类任务 参数为auc
	'n_estimators': 2000,								# 训练次数
	'eta': 0.1,											# 学习率
	'alpha': 0.1269124823585012,						# L1正则化 防止过拟合 可进行调整
	'subsample': 0.8345882521794742,					# 样本采集比率
	'colsample_bytree': 0.44270196445757065,			# 特征采样比例
	'max_depth': 15,
	'tree_method': 'hist',								# 使用直方图方法加快训练，适合大数据集
	'min_child_weight': 8,								# 深度较大的树中，确保叶子节点有足够的样本
	'gamma': 1.308021832047589e-08,						# 不进行节点分裂的最小损失下降，这可能让树更自由地生长
	'max_bin': 50000,									# 提高分割的精度，在使用hist方法时，注意是否有足够的内存支持
	'n_jobs': -1,										# 使用所有CPU核心
	'device': 'cuda',									# N卡提供的cuda工具，适用于windows及其含有GPU的Linux机器 Mac中的内存和显卡内存为公用内存 32
	'enable_categorical': True,							# 处理类别变量，适用于数据中有类别特征的情况，需要确保输入数据正确预处理。
	'early_stopping_rounds': 50,						# 早停轮数
	参数调试 超参数 / 也可以使用`蒸馏`
	https://blog.csdn.net/qq_43510916/article/details/113794486
	https://www.biaodianfu.com/optuna-xgboost.html
	https://developer.volcengine.com/articles/7436341741353238578
	https://zhuanlan.zhihu.com/p/637195017
	全连接神经网络参数调整
	https://blog.csdn.net/weixin_43114209/article/details/145157863
"""

xgb_params = {
	'eval_metric': 'auc',
	'n_estimators': 2000,
	'eta': 0.1,
	'alpha': 0.1269124823585012,
	'subsample': 0.8345882521794742,
	'colsample_bytree': 0.44270196445757065,
	'max_depth': 15,
	'tree_method': 'hist',
	'min_child_weight': 8,
	'gamma': 1.308021832047589e-08,
	'max_bin': 50000,
	'n_jobs': -1,
	'device': 'cuda',
	'enable_categorical': True,
	'early_stopping_rounds': 50,
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
y_test_prob = xgb_clf.get_booster().predict(xgb.DMatrix(X_test, enable_categorical=True),
											iteration_range=(0, xgb_clf.best_iteration + 1))
print(f"AUC: {roc_auc_score(y_test, y_test_prob):.6f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred), end="\n\n")

ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred), ).plot()
plt.title("Confusion Matrix", size=28)
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

plt.plot(fpr, tpr, lw=2, label=f"ROC Curve (Area= {roc_auc_score(y_test, y_test_prob):.6f})")
plt.plot([0, 1], [0, 1], 'k:')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Reciever Operating Characteristics(ROC) curve")
plt.legend()
plt.show()

# 获取基本的预测结果
test_pred_base = xgb_clf.get_booster().predict(xgb.DMatrix(test, enable_categorical=True),
											   iteration_range=(0, xgb_clf.best_iteration + 1))

# 两次判断进行数据拟合
# Case 1 欠采样
rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X, y)
X_train_rus, X_test_rus, y_train_rus, y_test_rus = train_test_split(X_rus, y_rus, test_size=0.2, shuffle=True,
																	random_state=42, stratify=y_rus)
xgb_clf_rus = xgb.XGBClassifier(**xgb_params)
xgb_clf_rus = xgb_clf_rus.fit(
	X_train_rus,
	y_train_rus,
	eval_set=[(X_test_rus, y_test_rus)],
	verbose=500
)
test_pred_rus = xgb_clf_rus.get_booster().predict(xgb.DMatrix(test, enable_categorical=True),
												  iteration_range=(0, xgb_clf_rus.best_iteration + 1))

# Case 2 过采样
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(X, y)
X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(X_ros, y_ros, test_size=0.2, shuffle=True,
																	random_state=42, stratify=y_ros)
xgb_clf_ros = xgb.XGBClassifier(**xgb_params)
xgb_clf_ros = xgb_clf_ros.fit(
	X_train_ros,
	y_train_ros,
	eval_set=[(X_test_ros, y_test_ros)],
	verbose=500
)

test_pred_ros = xgb_clf_ros.get_booster().predict(xgb.DMatrix(test, enable_categorical=True),
												  iteration_range=(0, xgb_clf_ros.best_iteration + 1))
sub = pd.DataFrame({
	'id': test.index,
	'Response': test_pred_ros
})

# PS 训练AUC为0.855680
print_fmt("result", sub)

# result output
pd.DataFrame({
	'id': test.index,
	'Response': test_pred_base
}).to_csv('submission_base.csv', index=False)

pd.DataFrame({
	'id': test.index,
	'Response': test_pred_rus
}).to_csv('submission_rus.csv', index=False)

# pd.DataFrame({
# 	'id': test.index,
# 	'Response': np.mean([test_pred_base, test_pred_rus, test_pred_ros], axis=0)
# }).to_csv('submission_master.csv', index=False)
