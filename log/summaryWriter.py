import statistics

class Summarize:
    def __init__(self, metrics, path):
        self.metrics = [float(i) for i in metrics]
        self.path = str(path)
        self.mean = statistics.mean(self.metrics)
        self.stddev = statistics.stdev(self.metrics)

    def writeSummary(self):
        f = open(self.path + "\\summary.log",'w')
        text = str(self.mean) + "±" + str(self.stddev)
        f.write(text)
        f.close()
