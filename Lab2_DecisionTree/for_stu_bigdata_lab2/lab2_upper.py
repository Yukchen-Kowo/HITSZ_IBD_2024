import pandas as pd
import numpy as np
import random

# 设置随机种子
random.seed(42)
np.random.seed(42)

# 0. 数据读取
columns = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'maritalStatus', 'occupation',
           'relationship', 'race', 'sex', 'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry', 'income']
df_train_set = pd.read_csv('./adult.data', names=columns)
df_test_set = pd.read_csv('./adult.test', names=columns, skiprows=1)

# 1. 数据预处理

# 删除不必要的列
df_train_set.drop(['fnlwgt'], axis=1, inplace=True)
df_test_set.drop(['fnlwgt'], axis=1, inplace=True)

# 替换缺失值
df_train_set.replace(' ?', np.nan, inplace=True)
df_test_set.replace(' ?', np.nan, inplace=True)

# 删除缺失值
df_train_set.dropna(inplace=True)
df_test_set.dropna(inplace=True)

# 去除首尾空格
for col in df_train_set.columns:
    if df_train_set[col].dtype == 'object':
        df_train_set[col] = df_train_set[col].str.strip()
for col in df_test_set.columns:
    if df_test_set[col].dtype == 'object':
        df_test_set[col] = df_test_set[col].str.strip()

# 处理目标变量
df_train_set['income'] = df_train_set['income'].map({'<=50K': 0, '>50K': 1})
df_test_set['income'] = df_test_set['income'].str.replace('.', '').map({'<=50K': 0, '>50K': 1})

# 连续型变量处理（标准化）
continuous_cols = ['age', 'educationNum', 'capitalGain', 'capitalLoss', 'hoursPerWeek']
for col in continuous_cols:
    mean_val = df_train_set[col].mean()
    std_val = df_train_set[col].std()
    df_train_set[col] = (df_train_set[col] - mean_val) / std_val
    df_test_set[col] = (df_test_set[col] - mean_val) / std_val  # 使用训练集的均值和标准差

# 合并训练集和测试集，方便对分类变量进行独热编码
df_train_set['source'] = 'train'
df_test_set['source'] = 'test'
df_combined = pd.concat([df_train_set, df_test_set], ignore_index=True)

# 对分类变量进行独热编码
categorical_cols = ['workclass', 'education', 'maritalStatus', 'occupation',
                    'relationship', 'race', 'sex', 'nativeCountry']
df_combined = pd.get_dummies(df_combined, columns=categorical_cols)

# 将数据集拆分回训练集和测试集
df_train_processed = df_combined[df_combined['source'] == 'train'].drop(['source'], axis=1)
df_test_processed = df_combined[df_combined['source'] == 'test'].drop(['source'], axis=1)

# 确保训练集和测试集的特征列一致
df_test_processed = df_test_processed.reindex(columns=df_train_processed.columns, fill_value=0)

# 2. 决策树实现

def calc_gini(df):
    labels = df['income']
    label_counts = labels.value_counts()
    total = len(labels)
    gini = 1.0 - sum((count / total) ** 2 for count in label_counts)
    return gini

def choose_best_feature_to_split(df):
    best_gini = float('inf')
    best_feature = None
    best_splits = None

    features = df.columns.drop('income')
    for feature in features:
        left_df = df[df[feature] == 1]
        right_df = df[df[feature] == 0]
        if len(left_df) == 0 or len(right_df) == 0:
            continue  # 跳过无效划分

        total_instances = len(df)
        weight_left = len(left_df) / total_instances
        weight_right = len(right_df) / total_instances
        gini_left = calc_gini(left_df)
        gini_right = calc_gini(right_df)
        gini_split = weight_left * gini_left + weight_right * gini_right

        if gini_split < best_gini:
            best_gini = gini_split
            best_feature = feature
            best_splits = (left_df, right_df)

    return best_feature, best_splits, best_gini

def build_decision_tree(df, max_depth=None, min_samples_split=2, depth=0):
    print(f"构建深度为 {depth} 的决策树，样本数：{len(df)}")
    labels = df['income']
    # 基础情况1：如果所有标签都相同，返回该标签
    if len(labels.unique()) == 1:
        return {'label': labels.iloc[0]}
    # 预剪枝条件
    if max_depth is not None and depth >= max_depth:
        majority_label = labels.value_counts().idxmax()
        return {'label': majority_label}
    if len(df) < min_samples_split:
        majority_label = labels.value_counts().idxmax()
        return {'label': majority_label}
    # 选择最好的划分特征
    best_feature, best_splits, best_gini = choose_best_feature_to_split(df)
    if best_feature is None:
        majority_label = labels.value_counts().idxmax()
        return {'label': majority_label}
    feature_name = best_feature
    # 构建子树
    left_df, right_df = best_splits

    # 调试信息
    print(f"在深度 {depth} 处划分特征 '{feature_name}'，基尼指数 {best_gini}")

    left_subtree = build_decision_tree(left_df, max_depth, min_samples_split, depth + 1)
    right_subtree = build_decision_tree(right_df, max_depth, min_samples_split, depth + 1)

    # 返回树
    return {'feature_name': feature_name,
            'left': left_subtree,
            'right': right_subtree}


def classify(cart, df_row):
    if 'label' in cart:
        return cart['label']
    else:
        feature_name = cart['feature_name']
        if feature_name not in df_row:
            # 如果特征缺失，默认走左子树
            return classify(cart['left'], df_row)
        if df_row[feature_name] == 1:
            return classify(cart['left'], df_row)
        else:
            return classify(cart['right'], df_row)

def predict(cart, df):
    pred_list = []
    for i in range(len(df)):
        df_row = df.iloc[i]
        pred_label = classify(cart, df_row)
        pred_list.append(pred_label)
    return pred_list

def calc_acc(pred_list, test_list):
    pred = np.array(pred_list)
    test = np.array(test_list)
    acc = np.sum(pred == test) / len(test)
    return acc

def cross_validate(df, k=5, max_depth=None, min_samples_split=2):
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    fold_size = len(df) // k
    acc_list = []
    for i in range(k):
        val_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)
        train_df = df.iloc[train_indices]
        val_df = df.iloc[val_indices]
        cart = build_decision_tree(train_df, max_depth=max_depth, min_samples_split=min_samples_split)
        pred_list = predict(cart, val_df)
        acc = calc_acc(pred_list, val_df['income'].to_numpy())
        acc_list.append(acc)
    mean_acc = np.mean(acc_list)
    return mean_acc

# 3. 训练和评估

params = [
    {'max_depth': None, 'min_samples_split': 2},
    {'max_depth': 10, 'min_samples_split': 2},
    {'max_depth': 10, 'min_samples_split': 5},
    {'max_depth': 15, 'min_samples_split': 10},
    {'max_depth': 5, 'min_samples_split': 5},
    {'max_depth': 8, 'min_samples_split': 10}
    # 可以继续添加更多参数组合
]

best_acc = 0
best_params = None
for param in params:
    acc = cross_validate(df_train_processed, k=5, max_depth=param['max_depth'],
                         min_samples_split=param['min_samples_split'])
    print(f"参数 {param} 下的交叉验证准确率为: {acc}")
    if acc > best_acc:
        best_acc = acc
        best_params = param

print(f"最佳参数为 {best_params}，交叉验证准确率为 {best_acc}")

# 使用最佳参数训练最终模型
cart = build_decision_tree(df_train_processed, max_depth=best_params['max_depth'],
                           min_samples_split=best_params['min_samples_split'])

# 在测试集上评估模型
pred_list = predict(cart, df_test_processed)
test_acc = calc_acc(pred_list, df_test_processed['income'].to_numpy())
print(f"在测试集上的准确率为: {test_acc}")

# 将预测结果输出到新的.csv文件中
df_test_processed['prediction'] = pred_list
df_test_processed.to_csv('test_predictions.csv', index=False)
print("预测结果已保存到 test_predictions.csv 文件中。")
