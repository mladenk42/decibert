from utils import AMIUtils
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import time
from amisvm import AMISvm

model_type = "SVM"


ut = AMIUtils()
df = ut.load_all_meetings_as_df()



def utterance_level_eval(pred_df, gold_df):
    for eval_class in ["I", "RP", "RR", "A"]:
        preds = pred_df["pred_" + eval_class]
        gold = gold_df["class_" + eval_class]
        P = precision_score(y_true = gold, y_pred = preds, pos_label = 1)
        R = recall_score(y_true = gold, y_pred = preds, pos_label = 1)
        F1 = f1_score(y_true = gold, y_pred = preds, pos_label = 1)
        print("%s --> P:%.3f   R:%.3f   F1:%.3f" % (eval_class, P, R, F1))

if model_type == "SVM":
    model = AMISvm()

# crossvalidation

start_time = time.time()

pred_dfs, gold_dfs = [], []
for mid in sorted(set(df.meeting_id), key = lambda x: x):
    print("Working on " + str(mid))
    train_set = df[df.meeting_id != mid]
    test_set = df[df.meeting_id == mid]
    model.fit(train_set)
    pred_dfs.append(model.predict(test_set))
    gold_dfs.append(test_set[["class_I", "class_RP", "class_RR", "class_A", "timestamp"]])
    break

print("Preds")
print(pred_dfs[0].head())
print("Gold")
print(gold_dfs[0].head())

utterance_level_eval(pd.concat(pred_dfs), pd.concat(gold_dfs))

print("Time needed was %s" % (time.time() - start_time))





