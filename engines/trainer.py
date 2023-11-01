class AbstractTrainer:

    def __init__(self, hyp, opt, device, callbacks):
        self.hyp = hyp
        self.device = device
        self.callbacks = callbacks
        self.opt = opt


    def train(self):
        pass


    def _do_train(self):
        pass

    def optimizer_step(self):
        pass


    def _process_batch(self, batch):
        pass

    def get_dataset(self, data):
        pass

    def save_model(self):
        pass

    def _setup_train(self):
        pass

    def _setup_ddp(self):
        pass


    def _setup_logger(self):
        pass