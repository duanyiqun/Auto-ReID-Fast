### By Yiqun Duan Jully30 2019

from __future__ import print_function, division

import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
version =  torch.__version__

import time
import utils.distributed as dist
## visualization
from tensorboardX import SummaryWriter
import yaml
import argparse
from utils.configurations import visualize_configurations, transfer_txt
from utils import loggers

from data import base_dataset
from models import baseline_cls, optimizers, losses
import utils.metrics as metrics

try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
    #fp16 = True
except:
    print('\nNo apex supports, using default setting in pytorch {} \n'.format(version))

######## solve multi-thread crash in IDEs #########
import multiprocessing
multiprocessing.set_start_method('spawn', True)

parser = argparse.ArgumentParser(description='Re-Implementation of Darts Based Partial Aware People Re-ID')
parser.add_argument('--config', default='configs/baseline_classification_PCB.yaml')
parser.add_argument("--verbose", default=False, help='whether verbose each stage')
parser.add_argument('--port', default=10530, type=int, help='port of server')
parser.add_argument('--distributed', default=False, help='switch to distributed training on slurm')
#parser.add_argument('--world-size', default=1, type=int)
#parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--model_dir', type=str)
parser.add_argument('--resume', default=False, help='resume')
#parser.add_argument('--test', dest='evaluate', action='store_true',help='evaluate model on validation set')
parser.add_argument('--fix_gpu_id', default=False, help='for extreme condition, some are not working')
parser.add_argument('--fp16', default= False, help='whether use apex quantization')
args = parser.parse_args()

if args.fix_gpu_id == False and torch.cuda.is_available():
    device = torch.device("cuda") 
    use_gpu = True
    try:
        if len(args.fix_gpu_id)>0:
            torch.cuda.set_device(args.fix_gpu_id[0])
    except:
        print('Not fixing GPU ids...')
else:
    device = torch.device("cpu")

#####################################################################
### history for draw graph

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

best_top1 = 0

######################################################################
# Save model
#---------------------------
def save_network(args, network, epoch_label, top1, isbest= False):
    if isbest:
        save_filename = 'best.pth'
    else:
        save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(args.checkpoint, args.task_name,save_filename)
    if not os.path.isdir(os.path.join(args.checkpoint, args.task_name)):
        os.makedirs(os.path.join(args.checkpoint, args.task_name))
    checkpoint = {}
    checkpoint['network'] = network.cpu().state_dict()
    checkpoint['epoch'] = epoch_label
    checkpoint['top1'] = top1
    torch.save(checkpoint, save_path)

def train(args, train_loader, valid_loader, model, woptimizer, lr_scheduler, epoch=0, criterion= False):
    print('-------------------training_start at epoch {}---------------------'.format(epoch))
    top1 = metrics.AverageMeter()
    top5 = metrics.AverageMeter()
    top10 = metrics.AverageMeter()
    losses = metrics.AverageMeter()

    cur_step = epoch*len(train_loader)

    lr_scheduler.step()
    lr = lr_scheduler.get_lr()[0]

    if args.distributed:
        if rank == 0:
            writer.add_scalar('train/lr', lr, cur_step)
    else:
        writer.add_scalar('train/lr', lr, cur_step)
    
    model.train()
    
    running_loss = 0.0
    running_corrects = 0.0
    step = 0

    for samples, labels in train_loader:
        step = step+1
        now_batch_size,c,h,w = samples.shape
        if now_batch_size<args.batch_size: # skip the last batch
            continue
        
        if use_gpu:
            #samples = Variable(samples.cuda().detach())
            #labels = Variable(labels.cuda().detach())
            samples, labels = samples.to(device) , labels.to(device)
        else:
            samples, labels = Variable(samples), Variable(labels)
        
        model.to(device)
        woptimizer.zero_grad()
        logits = model(samples)
        
        if not args.PCB:
            _, preds = torch.max(logits.data, 1)
            loss = criterion(logits, labels)
        else:
            part = {}
            sm = nn.Softmax(dim=1)
            num_part = 6
            for i in range(num_part):
                part[i] = logits[i]

            score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5]) 
            _, preds = torch.max(score.data, 1)

            loss = criterion(part[0], labels)
            for i in range(num_part-1):
                loss += criterion(part[i+1], labels)
        
        if epoch<args.warm_epoch and args.warm_up: 
            warm_iteration = round(len(train_loader)/args.batch_size)*args.warm_epoch # first 5 epoch
            warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
            loss *= warm_up
        
        if args.fp16: # we use optimier to backward loss
            with amp.scale_loss(loss, woptimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if args.w_grad_clip != False:
            nn.utils.clip_grad_norm_(model.weights(), args.w_grad_clip)

        
        if args.distributed:
            dist.simple_sync.sync_grad_sum(model)
        
        woptimizer.step()

        if args.distributed:
            dist.simple_sync.sync_bn_stat(model)
        
        if args.PCB:
            prec1, prec5, prec10 = metrics.accuracy(score, labels, topk=(1, 5, 10))            
        else:
            prec1, prec5, prec10 = metrics.accuracy(logits, labels, topk=(1, 5, 10))

        if args.distributed:
            dist.simple_sync.allreducemean_list([loss, prec1, prec5, prec10])

        losses.update(loss.item(), samples.size(0))
        top1.update(prec1.item(), samples.size(0))
        top5.update(prec5.item(), samples.size(0))
        top10.update(prec10.item(), samples.size(0))

        running_loss += loss.item() * now_batch_size

        #y_loss['train'].append(losses)
        #y_err['train'].append(1.0-top1)

        if args.distributed:
            if rank == 0:
                if step % args.print_freq == 0 or step == len(train_loader)-1:
                    logger.info(
                        "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch+1, args.epochs, step, len(train_loader)-1, losses=losses,
                            top1=top1, top5=top5))

                writer.add_scalar('train/loss', loss.item(), cur_step)
                writer.add_scalar('train/top1', prec1.item(), cur_step)
                writer.add_scalar('train/top5', prec5.item(), cur_step)
        else:
            if step % args.print_freq == 0 or step == len(train_loader)-1:
                logger.info(
                    "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, args.epochs, step, len(train_loader)-1, losses=losses,
                        top1=top1, top5=top5))

            writer.add_scalar('train/loss', loss.item(), cur_step)
            writer.add_scalar('train/top1', prec1.item(), cur_step)
            writer.add_scalar('train/top5', prec5.item(), cur_step)
            writer.add_scalar('train/top10', prec10.item(), cur_step)

        cur_step += 1
    if args.distributed:
        if rank == 0:
            logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, args.epochs, top1.avg))
    else:
        logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, args.epochs, top1.avg))

    if args.distributed:
        if rank ==0:
            if epoch % args.forcesave ==0:
                save_network(args,model,epoch,top1)
    else:
        if epoch % args.forcesave ==0:
            save_network(args,model,epoch,top1)

    
def validate(args, valid_loader, model, epoch=0, criterion= False, cur_step = 0):
    print('-------------------validation_start at epoch {}---------------------'.format(epoch))
    top1 = metrics.AverageMeter()
    top5 = metrics.AverageMeter()
    top10 = metrics.AverageMeter()
    losses = metrics.AverageMeter()

    model.eval()
    model.to(device)
    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            if args.distributed:
                if N< int(args.batch_size // world_size):
                    continue
            else:   
                if N<args.batch_size: # skip the last batch
                    continue

            logits = model(X)


            if not args.PCB:
                _, preds = torch.max(logits.data, 1)
                loss = criterion(logits, y)
            else:
                part = {}
                sm = nn.Softmax(dim=1)
                num_part = 6
                for i in range(num_part):
                    part[i] = logits[i]

                score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                _, preds = torch.max(score.data, 1)

                loss = criterion(part[0], y)
                for i in range(num_part-1):
                    loss += criterion(part[i+1], y)
            
            if args.PCB:
                prec1, prec5, prec10 = metrics.accuracy(score, y, topk=(1, 5, 10))            
            else:
                prec1, prec5, prec10 = metrics.accuracy(logits, y, topk=(1, 5, 10))

            if args.distributed:
                dist.simple_sync.allreducemean_list([loss, prec1, prec5, prec10])

            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)
            top10.update(prec10.item(), N)

            if args.distributed:
                if rank == 0:
                    if step % args.print_freq == 0 or step == len(valid_loader)-1:
                        logger.info(
                            "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                            "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                                epoch+1, args.epochs, step, len(valid_loader)-1, losses=losses,
                                top1=top1, top5=top5))
            
            else:
                if step % args.print_freq == 0 or step == len(valid_loader)-1:
                    logger.info(
                        "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch+1, args.epochs, step, len(valid_loader)-1, losses=losses,
                            top1=top1, top5=top5))

    if args.distributed:
        if rank == 0:
            writer.add_scalar('val/loss', losses.avg, cur_step)
            writer.add_scalar('val/top1', top1.avg, cur_step)
            writer.add_scalar('val/top5', top5.avg, cur_step)
            writer.add_scalar('val/top10', top10.avg, cur_step)

            logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}, Prec@5 {:.4%}, Prec@10 {:.4%}".format(epoch+1, args.epochs, top1.avg, top5.avg, top10.avg))

    else:
        writer.add_scalar('val/loss', losses.avg, cur_step)
        writer.add_scalar('val/top1', top1.avg, cur_step)
        writer.add_scalar('val/top5', top5.avg, cur_step)
        writer.add_scalar('val/top10', top10.avg, cur_step)

        logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}, Prec@5 {:.4%}, Prec@10 {:.4%}".format(epoch+1, args.epochs, top1.avg, top5.avg, top10.avg))

    return top1.avg

    
    #optimizer, scheduler

def main():
    global args, use_gpu, writer, rank, logger, best_top1, world_size, rank
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    
    #######  visualize configs ######
    visualize_configurations(config)
    #######  set args ######
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    if args.verbose:
        print('Config parsing complete')
    
    
    #######  world initial ######
    if args.distributed:
        rank, world_size = dist.dist_init(args.port, 'nccl')
        if rank == 0:
            tbpath = os.path.join(args.logpath, 'tb', args.task_name)
            if os.path.isdir(tbpath):
                writer = SummaryWriter(log_dir=tbpath)
            else:
                os.makedirs(tbpath)
                writer = SummaryWriter(log_dir=tbpath)
            writer.add_text('config_infomation', transfer_txt(args))
            logger = loggers.get_logger(os.path.join(args.logpath, '{}.distlog'.format(args.task_name)))
            logger.info("Logger is set ")
            logger.info("Logger with distribution")
    else:

        tbpath = os.path.join(args.logpath, 'tb', args.task_name)
        if os.path.isdir(tbpath):
            writer = SummaryWriter(log_dir=tbpath)
        else:
            os.makedirs(tbpath)
            writer = SummaryWriter(log_dir=tbpath)
        writer.add_text('config_infomation', transfer_txt(args))
        logger = loggers.get_logger(os.path.join(args.logpath, '{}.log'.format(args.task_name)))
        logger.info("Logger is set ")
        logger.info("Logger without distribution")
    

    ######## initial random setting #######
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    ######## test data reading ########

    since = time.time()
    dataset_train_val = base_dataset.baseline_dataset(args) 
    train_loader , val_loader = dataset_train_val.get_loader()
    logger.info("Initializing dataset used {} basic time unit".format(time.time()-since))

    logger.info("The training classes labels length :  {}".format(len(dataset_train_val.train_classnames)))
    since = time.time()
    inputs, classes = next(iter(train_loader))
    logger.info('batch loading time example is {}'.format(time.time()-since))

    ######### Init model ############
    if args.model_name == 'resnet50_middle':
        model = baseline_cls.resnet50_middle(len(dataset_train_val.train_classnames), droprate=args.dropoutrate,
        pretrain=args.pretrain, return_f=args.reture_bottleneck_feature, return_mid= args.return_middle_level_feature)
    else:
        model = baseline_cls.PCB(len(dataset_train_val.train_classnames))
    

    #logger.info(model)
    if args.PCB:
        model = baseline_cls.PCB(len(dataset_train_val.train_classnames))
    
    

    ########## lauch training ###########

    woptimizer =  optimizers.get_optimizer(args, model)
    lr_schedular = optimizers.get_lr_scheduler(args, woptimizer)
    criterion = losses.get_loss(args)

    if args.resume != '' and os.path.isfile(args.resume):
        if args.distributed:
            if rank == 0:
                print('resuem from [%s]' % config.resume)
            checkpoint = torch.load(
                args.resume,
                map_location = 'cuda:%d' % torch.cuda.current_device()
            )
        else:
            print('resuem from [%s]' % config.resume)
            checkpoint = torch.load(config.resume,map_location = "cpu")

        model.load_state_dict(checkpoint['network'])
        #woptimizer.load_state_dict(checkpoint['optimizer'])
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        epoch_offset = checkpoint['epoch']
    else:
        epoch_offset = 0
    
    model.to(device)
    if args.distributed:
        dist.sync_state(model)

    if args.fp16:
        model, woptimizer = amp.initialize(model, woptimizer, opt_level = "O1")
    
    for epoch in range(epoch_offset, args.epochs):

        # train
        train(args,train_loader,val_loader,model, woptimizer, lr_schedular, epoch=epoch, criterion=criterion)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(args, val_loader, model, epoch= epoch, cur_step = cur_step, criterion= criterion)

        if args.distributed:    
            if rank ==0:
                if best_top1 < top1:
                    best_top1 = top1
                    save_network(args,model,epoch,top1, isbest=True)
                else:
                    if epoch % args.forcesave ==0:
                        save_network(args,model,epoch,top1)
                writer.add_scalar('val/best_top1', best_top1, cur_step)
        
        else:
            if best_top1 < top1:
                best_top1 = top1
                save_network(args,model,epoch,top1, isbest=True)
            else:
                if epoch % args.forcesave ==0:
                    save_network(args,model,epoch,top1)
            
            writer.add_scalar('val/best_top1', best_top1, cur_step)
            

        if args.distributed:
            if rank == 0:
                logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
                #logger.info("Best Genotype = {}".format(best_genotype))
        else:
            logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


if __name__ == '__main__':
    main()
