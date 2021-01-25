from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
from scipy.sparse import csr_matrix, vstack as sparse_vstack, hstack as sparse_hstack
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer

class AMISvm:
    def __init__(self):
        self.do_scaling = True
        
        self.do_feat_sel = False
        self.feat_sel_count = 500
        
        self.ngram_rng = (1,1)
        self.min_df = 3
        self.use_idf = True

        self.prev_context_len = 0
        self.next_context_len = 0

        self.use_utterance_feats = False

        self.transformer_model = SentenceTransformer("paraphrase-distilroberta-base-v1")
        #self.transformer_model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

    def _generate_feats(self, data, mode):
        # lexical feats
        #if mode == "train":
        #    self.tfidf_vect = TfidfVectorizer(ngram_range = self.ngram_rng, min_df = self.min_df, use_idf = self.use_idf)
        #    self.tfidf_vect.fit([x[1:-1] for x in list(data.text)]) # the x[1:-1] strips the initial and final [ and ] from the texts
        #feats = self.tfidf_vect.transform([x[1:-1] for x in list(data.text)])
         
        
        feats = self.transformer_model.encode([x[1:-1] for x in list(data.text)])
        feats = np.array(feats)

        if self.use_utterance_feats:
            # utterance feats
            ut_feats = np.zeros((data.shape[0], 3))

            current_mid = data.iloc[0,8]
            current_max_timestamp = max(data[data.meeting_id == current_mid].timestamp)
            for i in range(data.shape[0]):
                text = data.iloc[i,2][1:-1]
                timestamp = data.iloc[i,1]
                next_timestamp = data.iloc[i+1, 1] if (i+1 < data.shape[0] and data.iloc[i + 1,8] == data.iloc[i, 8])  else None 
                # first condition is for the end of the data frame (last utterance of the last meeting) second is on a breaking point between two meetings (happens for last utterance of every meeting)
                # without the second we would get 1853.2 as the last timestamp of meeting X and 0.0 as the first in meeting Y and the difference would be negative which messes up things down the line

                ut_feats[i,0] = len(text.split(" ")) # length in words
                ut_feats[i,1] = next_timestamp - timestamp if next_timestamp is not None else 2.0 # 2.0 is just an arbitray approximate value for the duration of the last utterance of each meeting 
                ut_feats[i,2] = timestamp / current_max_timestamp 

                if next_timestamp is None and i+1 < data.shape[0]: # this is a breaking point between meetings and we have to update some of the vals for the next iteration
                    current_mid = data.iloc[i+1,8]
                    current_max_timestamp = max(data[data.meeting_id == current_mid].timestamp)


            feats = csr_matrix(sparse_hstack([feats, csr_matrix(ut_feats)]))

        # expand all utterance level  feats to include feats of the prev and next utterances
        prev_context_feat_mats, next_context_feat_mats = [], []
        # prev context
        for offset in range(1, self.prev_context_len + 1):
            context_feats = feats[:-offset,:] 
            padding = csr_matrix(np.zeros((offset,feats.shape[1])))
            final = sparse_vstack((padding, context_feats))
            prev_context_feat_mats.append(final)

        # next context
        for offset in range(1, self.next_context_len + 1):
            context_feats = feats[offset:,:] 
            padding = csr_matrix(np.zeros((offset,feats.shape[1])))
            final = sparse_vstack((context_feats, padding))
            next_context_feat_mats.append(final)
        
        #feats = sparse_hstack([feats] + prev_context_feat_mats + next_context_feat_mats)
        
        if self.do_scaling:
          if mode == "train":
            self.scaler = StandardScaler(with_mean = False)
            self.scaler.fit(feats)
          feats = self.scaler.transform(feats)

        return feats
        

    def fit(self, train_data):
        X = self._generate_feats(train_data, mode = "train")
        self.models = {}
        if self.do_feat_sel:
            self.feat_selectors = {}

        for class_name in ["I","RP", "RR","A"]:
            #print("Fitting svm for class " + class_name + " ...")
            self.models[class_name]  = LogisticRegression(C = 10, class_weight = "balanced", max_iter = 100000)            
            #self.models[class_name]  = LinearSVC(C = 1, class_weight = "balanced")
            #self.models[class_name]  = MLPClassifier(hidden_layer_sizes = [5], alpha = 0.001, max_iter = 5000)

            y = list(train_data["class_" + class_name])

            if self.do_feat_sel:
              self.feat_selectors[class_name] = SelectKBest(chi2, k = self.feat_sel_count)
              self.feat_selectors[class_name].fit(X,y)
              X_featsel = self.feat_selectors[class_name].transform(X)
            else:
              X_featsel = X

            self.models[class_name].fit(X_featsel, y)


    def predict(self, test_data):
        X = self._generate_feats(test_data,  mode = "predict")
        pred_df = pd.DataFrame()

        for class_name in ["I","RP", "RR","A"]:
            #print("Prdicting with svm for class " + class_name + " ...")
            if self.do_feat_sel:
              X_featsel = self.feat_selectors[class_name].transform(X)
            else:
              X_featsel = X
                
            pred_df["pred_" + class_name] = self.models[class_name].predict(X_featsel)

        return pred_df 


