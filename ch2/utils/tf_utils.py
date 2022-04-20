import nni
from tensorflow.keras.callbacks import Callback


class TfNniIntermediateResult(Callback):

    def __init__(self, key):
        super().__init__()
        self.key = key

    def on_epoch_end(self, epoch, logs = None):
        nni.report_intermediate_result(logs[self.key])
