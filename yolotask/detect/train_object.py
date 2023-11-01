import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import lr_scheduler

from engines.trainer import AbstractTrainer
from models.task import Model
from utils.autobatch import check_train_batch_size
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download
from utils.general import LOGGER, colorstr, methods, check_suffix, init_seeds, check_dataset, intersect_dicts, \
    check_amp, check_img_size, one_cycle, labels_to_class_weights
from utils.loggers import Loggers
from utils.torch_utils import torch_distributed_zero_first, smart_optimizer, ModelEMA, smart_resume, smart_DDP

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None  # check_git_info()


class DetectTrainer(AbstractTrainer):
    def __init__(self, hyp, opt, device, callbacks):
        super().__init__(hyp, opt, device, callbacks)

        # Directories
        if not opt['info-only']:
            w = self.opt['save-dir'] / 'weights'
            (w.parent if self.opt['evolve'] else w).mkdir(parents=True, exist_ok=True)
            self.last, self.best = w / 'last.pt', w / 'best.pt'

        # Hyperparameters
        if isinstance(self.hyp, str):
            with open(self.hyp, errors='ignore') as f:
                self.hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        self.hyp['anchor_t'] = 5.0
        self.opt['hyp'] = self.hyp.copy()  # for saving hyps to checkpoints

        # Loggers
        self.data_dict = None
        if RANK in {-1, 0}:
            self._setup_logger()
            self.data_dict = self.loggers.remote_dataset

    def _setup_logger(self):
        self.loggers = Loggers(self.opt['save-dir'], self.opt['weights'], self.opt, self.hyp,
                               LOGGER)  # loggers instance

        # Register actions
        for k in methods(self.loggers):
            self.callbacks.register_action(k, callback=getattr(self.loggers, k))

        # # Process custom dataset artifact link
        # if self.opt['resume']:  # If resuming runs from remote artifact
        #     self.weights, self.epochs, self.hyp, self.batch_size = self.opt['weights'], self.opt['epochs'], \
        #                                                            self.opt['hyp'], self.opt['batch_size']

    def _setup_train(self, word_size):
        self.callbacks.run('on_pretrain_routine_start')

        # Config
        self.plots = not self.opt['evolve'] and not self.opt['noplots']  # create plots
        self.cuda = self.device.type != 'cpu'
        init_seeds(self.opt['seed'] + 1 + RANK, deterministic=self.opt['deterministic'])
        with torch_distributed_zero_first(LOCAL_RANK):
            self.data_dict = self.data_dict or check_dataset(self.opt['data'])  # check if None
        train_path, val_path = self.data_dict['train'], self.data_dict['val']
        self.nc = 1 if self.opt['single-cls'] else int(self.data_dict['nc'])  # number of classes
        self.names = {0: 'item'} if self.opt['single-cls'] and len(self.data_dict['names']) != 1 else self.data_dict[
            'names']  # class names
        is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

        # Model
        check_suffix(self.opt['weights'], '.pt')  # check weights
        pretrained = self.opt['weights'].endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(LOCAL_RANK):
                self.opt['weights'] = attempt_download(self.opt['weights'])  # download if not found locally
            self.ckpt = torch.load(self.opt['weights'],
                                   map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
            self.model = Model(self.opt['cfg'] or self.ckpt['model'].yaml, ch=3, nc=self.nc,
                               anchors=self.hyp.get('anchors')).to(self.device)  # create
            exclude = ['anchor'] if (self.opt['cfg'] or self.hyp.get('anchors')) and not self.opt[
                'resume'] else []  # exclude keys
            csd = self.ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, self.model.state_dict(), exclude=exclude)  # intersect
            self.model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(
                f"Transferred {len(csd)}/{len(self.model.state_dict())} items from {self.opt['weights']}")  # report
        else:
            self.model = Model(self.opt['cfg'], ch=3, nc=self.nc, anchors=self.hyp.get('anchors')).to(
                self.device)  # create
        if self.opt['info-only']:
            return False

        # amp check
        if not self.opt['use-amp']:
            self.amp = False
            LOGGER.info(f"{colorstr('AMP:')}AMP off")
        else:
            self.amp = check_amp(self.model)  # check AMP

        # Freeze
        freeze = [f'model.{x}.' for x in (
            self.opt['freeze'] if len(self.opt['freeze']) > 1 else range(self.opt['freeze'][0]))]  # layers to freeze
        for k, v in self.model.named_parameters():
            # v.requires_grad = True  # train all layers TODO: uncomment this line as in master
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in self.opt['freeze']):
                LOGGER.info(f'freezing {k}')
                v.requires_grad = False

        # Image size
        self.gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        self.imgsz = check_img_size(self.opt['imgsz'], self.gs, floor=self.gs * 2)  # verify imgsz is gs-multiple

        # Batch size
        if RANK == -1 and self.opt['batch-size'] == -1:  # single-GPU only, estimate best batch size
            self.opt['batch-size'] = check_train_batch_size(self.model, self.imgsz, self.amp)
            self.loggers.on_params_update({"batch_size": self.opt['batch-size']})

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / self.opt['batch-size']), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= self.opt['batch-size'] * accumulate / nbs  # scale weight_decay
        self.optimizer = smart_optimizer(self.model, self.opt['optimizer'], self.hyp['lr0'], self.hyp['momentum'],
                                         self.hyp['weight_decay'])

        # Scheduler
        if self.opt['cos-lr']:
            self.lf = one_cycle(1, self.hyp['lrf'], self.opt['epochs'])  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.opt['epochs']) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear

        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

        # EMA
        self.ema = ModelEMA(self.model) if RANK in {-1, 0} else None

        # Resume
        self.best_fitness, self.start_epoch = 0.0, 0
        if pretrained:
            if self.opt['resume']:
                self.best_fitness, self.start_epoch, self.epochs = smart_resume(self.ckpt, self.optimizer, self.ema,
                                                                                self.opt['weights'], self.opt['epochs'],
                                                                                self.opt['resume'])
            del self.ckpt, csd

        # DP mode
        if self.cuda and RANK == -1 and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                           'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if self.opt['sync-bn'] and self.cuda and RANK != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            LOGGER.info('Using SyncBatchNorm()')

        # Trainloader
        self.train_loader, self.dataset = create_dataloader(train_path,
                                                            self.imgsz,
                                                            self.opt['batch-size'] // WORLD_SIZE,
                                                            self.gs,
                                                            self.opt['single-cls'],
                                                            hyp=self.hyp,
                                                            augment=True,
                                                            cache=None if self.opt['cache'] == 'val' else self.opt[
                                                                'cache'],
                                                            rect=self.opt['rect'],
                                                            rank=LOCAL_RANK,
                                                            workers=self.opt['workers'],
                                                            image_weights=self.opt['image-weights'],
                                                            quad=self.opt['quad'],
                                                            prefix=colorstr('train: '),
                                                            shuffle=True,
                                                            min_items=self.opt['min-items'])
        self.labels = np.concatenate(self.dataset.labels, 0)
        mlc = int(self.labels[:, 0].max())  # max label class
        assert mlc < self.nc, f"Label class {mlc} exceeds nc={self.nc} in {self.opt['data']}. Possible class labels are 0-{self.nc - 1}"

        # Process 0
        if RANK in {-1, 0}:
            self.val_loader = create_dataloader(val_path,
                                                self.imgsz,
                                                self.opt['batch-size'] // WORLD_SIZE * 2,
                                                self.gs,
                                                self.opt['single-cls'],
                                                hyp=self.hyp,
                                                cache=None if self.opt['noval'] else self.opt['cache'],
                                                rect=True,
                                                rank=-1,
                                                workers=self.opt['workers'] * 2,
                                                pad=0.5,
                                                prefix=colorstr('val: '))[0]

            if not self.opt['resume']:
                # if not opt.noautoanchor:
                #     check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
                self.model.half().float()  # pre-reduce anchor precision

            self.callbacks.run('on_pretrain_routine_end', self.labels, self.names)




    def _setup_ddp(self):
        # DDP mode
        if self.cuda and RANK != -1:
            self.model = smart_DDP(self.model)

        # Model attributes
        # nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
        # hyp['box'] *= 3 / nl  # scale to layers
        # hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        # hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.hyp['label_smoothing'] = self.opt['label_smoothing']
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(self.device) * self.nc  # attach class weights
        self.model.names = self.names



    def _do_train(self):
        pass
