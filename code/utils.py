import pandas as pd

class AMIUtils:
    def __init__(self):
        pass

    def load_meeting_as_df(self, filename):
        print("Loading meeting --> " + filename)
        df = pd.read_csv("../data/" + filename, header = None, sep = "\t")
        df.columns = ["link1", "speaker_id", "timestamp", "text", "classes_orig", "link2"]
        df = df.drop("link1", axis = 1)
        df = df.drop("link2", axis = 1)
        
        df["speaker_id"] = df["speaker_id"].astype(str)
        df["timestamp"] = df["timestamp"].astype(float)
        df["text"] = df["text"].astype(str)
        df["classes_orig"] = df["classes_orig"].astype(str)

        def find_class(trigger, class_string):
            if class_string == "nan":
                return 0
            else:
                return 1 if trigger in class_string else 0

        df['class_I'] = df['classes_orig'].apply(lambda x: find_class("i", x))       
        df['class_RP'] = df['classes_orig'].apply(lambda x: find_class("s", x))       
        df['class_RR'] = df['classes_orig'].apply(lambda x: find_class("m", x))       
        df['class_A'] = df['classes_orig'].apply(lambda x: find_class("g", x))       
        
        df["meeting_id"] = filename.split("-")[0]
        
        #l = list(df['classes_orig'])
        #l = [x for x in l if x != "nan"]
        #print(l)
        #single = sum([1 if len(x)>1 else 0 for x in l])
        #print(single/len(l))
        return df

    def load_all_meetings_as_df(self):
        file_list = self.get_meeting_file_list()
        dataframes = [self.load_meeting_as_df(x) for x in file_list]
        return pd.concat(dataframes)

    def get_meeting_file_list(self):
       return ["ES2006c-mframpton-da-deepdecision.txt",
               "ES2009b-mframpton-da-deepdecision.txt",
               "ES2009c-raquel-da-deepdecision.txt",
               "ES2010b-pavani-da-deepdecision.txt",
               "ES2010c-raquel-da-deepdecision.txt",
               "ES2012c-raquel-da-deepdecision.txt",
               "ES2015b-raquel-da-deepdecision.txt",  
               "ES2015c-raquel-da-deepdecision.txt",
               "ES2016c-mframpton-da-deepdecision.txt",
               "IS1001c-pavani-da-deepdecision.txt",
               "IS1003c-pavani-da-deepdecision.txt",
               "IS1004c-pavani-da-deepdecision.txt",
               "IS1006c-pavani-da-deepdecision.txt",
               "IS1008c-pavani-da-deepdecision.txt",
               "TS3004b-mframpton-da-deepdecision.txt",
               "TS3005b-pavani-da-deepdecision.txt",
               "TS3005c-pavani-da-deepdecision.txt"]


    


