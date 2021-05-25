import os
import torch

class Logger:
# creating a constructor creates a directory path 
    def __init__(self, save_path, dataset_name, model_name, run_id):
        self.cwd = os.getcwd()
        self.save_path = self.cwd + "\\" + str(save_path)
        self.dataset_name = str(dataset_name)
        self.model_name = str(model_name)
        self.run_id = str(run_id)
        self.path = self.save_path + "\\" + self.dataset_name + "\\" + self.model_name
        self.logfile = None

    def createDir(self):
        os.makedirs(self.path, exist_ok=True)

    def cleanup(self):
        pass

    def start(self):
        self.logfile = open(self.path + "\\" + self.run_id + ".log", "w")

    def writelog(self, text):
        text = str(text)
        self.logfile = open(self.path + "\\" + self.run_id + ".log", "a")
        self.logfile.write(text)
        self.logfile.write("\n")
        self.logfile.close()

    def saveModel(self, model):
        model_file = self.path + "\\" + self.model_name + ".pth"
        torch.save(model.state_dict(),model_file)