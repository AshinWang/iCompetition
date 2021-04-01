# [全球人工智能技术创新大赛 赛道三: 小布助手对话短文本语义匹配](https://tianchi.aliyun.com/competition/entrance/531851/introduction)



## 题意

```python
肖战的粉丝叫什么名字 肖战的粉丝叫什么 1

王者荣耀里面打野谁最厉害 王者荣耀什么英雄最好玩 0

我想换个手机 我要换手机 1

我是张睿 我想张睿 0

不想 不想说 0
```

训练数据包含输入query-pair，以及对应的真值。初赛训练样本10万，复赛训练样本30万，这份数据主要用于参赛队伍训练模型，为确保数据的高质量，每一个样本的真值都有进行人工标注校验。每行为一个训练样本，由query-pair和真值组成，每行格式如下：

- query-pair格式：query以中文为主，中间可能带有少量英文单词（如英文缩写、品牌词、设备型号等），采用UTF-8编码，未分词，两个query之间使用\t分割。
- 真值：真值可为0或1，其中1代表query-pair语义相匹配，0则代表不匹配，真值与query-pair之间也用\t分割。



## 方案

根据官方提供的 [Baseline](https://aliyuntianchiresult.cn-hangzhou.oss.aliyun-inc.com/file/race/documents/531851/baseline_tfidf_lr_0_75.tar.gz?Expires=1617370895&OSSAccessKeyId=LTAILBoOl5drlflK&Signature=RMt4zHoNaWbhA1bp4rYvIe3oYDk%3D&response-content-disposition=attachment%3B%20) ，添加特征，将此比赛当作结构化数据进行二分类，提交分数在 0.8 左右。

### 添加特征

```python
#  字符串拆分城数字
def str2char():
    txt = df['text'].apply(lambda x: list(x.split()))
    x = []   
    for i in (range(0, len(txt))):
        x.append(txt[i])
        y = x[i]   
        for j in range(len(y)):
            df['str2char_{}'.format(j)] = 0

    for i in (range(0, len(txt))):
        x.append(txt[i])
        y = x[i]       
        for value, idx in zip(y, range(len(y))):
            df['str2char_{}'.format(idx)][i] = value
str2char()

#   q1、q2 字符串中的相同字符个数
def dup_num():
    a = df['q1'].apply(lambda x: (x.split()))
    b = df['q2'].apply(lambda x: (x.split()))
    x = []
    for i in range(len(a)):
        x.append(len(set(a[i]) & set(b[i])))
    df['dup_num']=pd.DataFrame(x)
    return df['dup_num']
df['dup_num'] = dup_num()

#   q1、q2 字符串中的相同字符
def dup_chr():
    a = df['q1'].apply(lambda x: list(x.split()))
    b = df['q2'].apply(lambda x: list(x.split()))
    x = []   
    for i in (range(0, len(a))):
        x.append(set(a[i]) & set(b[i]))
        y = x[i]       
        for  idx in  range(len(y)):
            df['dup_chr_{}'.format(idx)] = 0

    for i in (range(0, len(a))):
        x.append(set(a[i]) & set(b[i]))
        y = x[i]       
        for value, idx in zip(y, range(len(y))):
            df['dup_chr_{}'.format(idx)][i] = value
dup_chr()

# 长度
df['q1_len']   = df['q1'].apply(lambda x: len(x.split(' '))) 
df['q2_len']   = df['q2'].apply(lambda x: len(x.split(' '))) 
df['text_len'] = df['text'].apply(lambda x: len(x.split(' '))) 

# q1 q2长度差
df['sub_q1q2'] = df['q1_len'] - df['q2_len']
df['sub_q2q1'] = df['q2_len'] - df['q1_len']

# 出现的最大值
df['max_q1']     = df['q1'].apply(lambda x: max(list(map(int, x.split()))))
df['max_q2']     = df['q2'].apply(lambda x: max(list(map(int, x.split()))))
df['max_text'] = df['text'].apply(lambda x: max(list(map(int, x.split()))))

# 出现的最小值
df['min_q1']     = df['q1'].apply(lambda x: min(list(map(int, x.split()))))
df['min_q2']     = df['q2'].apply(lambda x: min(list(map(int, x.split()))))
df['min_text'] = df['text'].apply(lambda x: min(list(map(int, x.split()))))

# max-min
df['max_min_q1']     = df['max_q1'] - df['min_q1']
df['max_min_q2']     = df['max_q2'] - df['min_q2']
df['max_min_text'] = df['max_text'] - df['min_text']

# mean值
df['mean_q1']     = df['q1'].apply(lambda x: mean(list(map(int, x.split()))))
df['mean_q2']     = df['q2'].apply(lambda x: mean(list(map(int, x.split()))))
df['mean_text'] = df['text'].apply(lambda x: mean(list(map(int, x.split()))))

# std值
df['std_q1']     = df['q1'].apply(lambda x: std(list(map(int, x.split()))))
df['std_q2']     = df['q2'].apply(lambda x: std(list(map(int, x.split()))))
df['std_text'] = df['text'].apply(lambda x: std(list(map(int, x.split()))))

# freq 出现的最大频率
df['freq_q1']     = df['q1'].apply(lambda x: max(((Counter(list(map(int, x.split()))))).values()))
df['freq_q2']     = df['q2'].apply(lambda x: max(((Counter(list(map(int, x.split()))))).values()))
df['freq_text'] = df['text'].apply(lambda x: max(((Counter(list(map(int, x.split()))))).values()))
```



### lgbm

```python
import lightgbm as lgb

scores = []

nfold = 5
kf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)

lr_oof = np.zeros((len(df_train), 2))
lr_predictions = np.zeros((len(df_test), 2))

i = 0
for train_index, valid_index in kf.split(train_df, label):
    print("\nFold {}".format(i + 1))
    X_train, label_train = train_df[train_index], label[train_index]
    X_valid, label_valid = train_df[valid_index], label[valid_index]


    model = lgb.LGBMClassifier(num_leaves=31, max_depth=13, n_estimators=100000, learning_rate=0.05, verbose=-1, metric='auc')
    model.fit(X_train, label_train, eval_set = [(X_valid, label_valid)], early_stopping_rounds=300, verbose=600)


    lr_oof[valid_index] = model.predict_proba(X_valid,)
    scores.append(roc_auc_score(label_valid, lr_oof[valid_index][:,1]))
    
    lr_predictions += model.predict_proba(test_df) / nfold
    i += 1
    print(scores)
    
print(np.mean(scores))
```

