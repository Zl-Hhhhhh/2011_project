## 数据处理说明

数据预处理代码位于 `project.ipynb` 的 Part1，输入文件为 `data/data.csv`（分隔符 `;`）。

### 1) 两种模式（task_mode）

- `early`：用于早期预警，删除可能泄漏未来学业表现的信息字段（如包含 `approved`、`grade`、`evaluations`、`credited` 等关键词的列）。
- `full`：保留全部原始特征，用于完整信息建模。

做 Early Warning 场景时优先使用 `early`。意思是还不清楚与`target`高度相关的`grade`等数据时进行预测。

### 如果需要 full 数据，怎么修改

在 `project.ipynb` 的 Part1 中，把：

`task_mode = "early"`

改为：

`task_mode = "full"`

然后重新运行 part1（Data Preprocessing）以及后续依赖该结果的单元。此时不会删除泄漏关键词相关列，输出的 `X_processed` 即为 full 模式特征。

### 2) 特征编码规则

- 先做基础清洗：列名去空格；缺失值用中位数/众数填补；并新增缺失指示列 `<col>_missing`。
- 将“编码后的类别整数列”（如 `Marital status`、`Course`、`Gender` 等）转为 `category` 类型。
- 将仅包含 0/1 的数值列转为 `bool`。
- 最后对特征做 One-Hot 编码：`X_processed = pd.get_dummies(X, drop_first=False, dtype=int)`。

### 2.1) 标准化结论

- Part1 的尺度检查显示：非二值数值特征尺度差异较大（`scale_ratio = 20.96`），因此会生成 `X_standardized`（仅标准化非二值特征，保留 0/1 与 one-hot 列原值）。

### 3) 后续可直接使用的数据

- `X_processed`：最终特征矩阵（已完成清洗 + 类型处理 + One-Hot）。
- `X_standardized`：仅对非二值数值特征做标准化后的特征矩阵（更适合距离类/线性模型训练），原有`X_processed`更适合tree。
- `y`：原始目标标签（`Target` 列）。
- `y_processed`：当目标为非数值时的 One-Hot 标签；若目标本身是数值，则等同于 `y`。
- `target_pct`：各类别占比（用于判断类别不平衡）。