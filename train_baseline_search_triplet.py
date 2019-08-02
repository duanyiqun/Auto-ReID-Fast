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

from data import tripletsample_dataset as base_dataset
from models import baseline_cls, optimizers, losses

from models.DARTS.archetect import Architect
from models.DARTS.search_cnn import SearchCNNController

import utils.metrics as metrics

try:
    from utils.visualization import plot
except:
    print('\nNo graphic visualization supports...')

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
parser.add_argument('--config', default='configs/baseline_classification_DARTS_distributed.yaml')
parser.add_argument("--verbose", default=False, help='whether verbose each stage')
parser.add_argument('--port', default=10530, type=int, help='port of server')
parser.add_argument('--distributed', default=False, help='switch to distributed training on slurm')
#parser.add_argument('--world-size', default=1, type=int)
#parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--resume', default=False, help='resume')

parser.add_argument('--fix_gpu_id', default=False, help='for extreme condition, some are not working')
parser.add_argument('--sync_grad_sum', default=True, help='sychronize sum or mean')
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




def train(args, train_loader, valid_loader, model, architect,  w_optim, alpha_optim, lr_scheduler, epoch=0):
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
    #step = 0
    model.to(device)

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        #step = step+1
        now_batch_size,c,h,w = trn_X.shape
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)

        if args.distributed:
            if now_batch_size< int(args.batch_size // world_size):
                continue
        else:   
            if now_batch_size<args.batch_size: # skip the last batch
                continue

        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()


        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        
        # gradient clipping\
        if args.w_grad_clip != False:
            nn.utils.clip_grad_norm_(model.weights(), args.w_grad_clip)

        if args.distributed:
            if args.sync_grad_sum:
                dist.sync_grad_sum(model)
            else:
                dist.sync_grad_mean(model)
        

        w_optim.step()

        if args.distributed:
            dist.sync_bn_stat(model)

        prec1, prec5, prec10 = metrics.accuracy(logits, trn_y, topk=(1, 5, 10))

        if args.distributed:
            dist.simple_sync.allreducemean_list([loss, prec1, prec5, prec10])

        losses.update(loss.item(), now_batch_size)
        top1.update(prec1.item(), now_batch_size)
        top5.update(prec5.item(), now_batch_size)
        top10.update(prec10.item(), now_batch_size)
        
        #running_loss += loss.item() * now_batch_size

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
                writer.add_scalar('train/top10', prec10.item(), cur_step)
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
    
    if epoch % args.forcesave ==0:
        save_network(args,model,epoch,top1)
        
    
        

def validate(args, valid_loader, model, epoch=0, cur_step = 0):
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

            ### 必须加分布式判断，否则validation跳过一直为真。
            if args.distributed:
                if N< int(args.batch_size // world_size):
                    continue
            else:   
                if N<args.batch_size: # skip the last batch
                    continue

            logits = model(X)
            loss = model.criterion(logits, y)

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
        logger = loggers.get_logger(os.path.join(args.logpath, '{}.distlog'.format(args.task_name)))
        if rank == 0:
            tbpath = os.path.join(args.logpath, 'tb', args.task_name)
            if os.path.isdir(tbpath):
                writer = SummaryWriter(log_dir=tbpath)
            else:
                os.makedirs(tbpath)
                writer = SummaryWriter(log_dir=tbpath)
            writer.add_text('config_infomation', transfer_txt(args))
            
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
    #woptimizer =  optimizers.get_optimizer(args, model)
    #lr_schedular = optimizers.get_lr_scheduler(args, woptimizer)
    criterion = losses.get_loss(args)

    criterion.to(device)

    if args.model_name == 'Darts_normal':
        model = SearchCNNController(args.input_channels, args.init_channels, len(dataset_train_val.train_classnames), args.Search_layers, criterion)
    else:
        model = SearchCNNController(args.input_channels, args.init_channels, len(dataset_train_val.train_classnames), args.Search_layers, criterion)
    
    model = model.to(device)
    if args.distributed:
        dist.sync_state(model)
    
    w_optim = torch.optim.SGD(model.weights(), args.w_lr, momentum=args.w_momentum,
                              weight_decay=args.w_weight_decay)
        
    alpha_optim = torch.optim.Adam(model.alphas(), args.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=args.alpha_weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, args.epochs, eta_min=args.w_lr_min)
    architect = Architect(model, args.w_momentum, args.w_weight_decay, args)


    ########## lauch training ###########

    

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

    if args.fp16:
        model, w_optim = amp.initialize(model, w_optim, opt_level = "O1")
    
    for epoch in range(epoch_offset, args.epochs):
        if args.distributed:
            if rank == 0:
                model.print_alphas(logger)
        else:
            model.print_alphas(logger)

        # train
        if epoch % args.real_val_freq == 0:
            train(args,train_loader,val_loader,model,architect, w_optim, alpha_optim, lr_scheduler, epoch=epoch)
        else:
            train(args,train_loader,train_loader,model,architect, w_optim, alpha_optim, lr_scheduler, epoch=epoch)            
        # validation
        cur_step = (epoch+1) * len(train_loader)

        top1 = validate(args, val_loader, model, epoch= epoch, cur_step = cur_step)

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

        genotype = model.genotype()
        
        if args.distributed:

            if rank == 0:
                logger.info("genotype = {}".format(genotype))
                
                if args.plot_path != False:
            
                    plot_path = os.path.join(args.plot_path, args.task_name, "EP{:02d}".format(epoch+1))
                    if not os.path.isdir(os.path.join(args.plot_path, args.task_name)):
                        os.makedirs(os.path.join(args.plot_path, args.task_name))
                    caption = "Epoch {}".format(epoch+1)
                    plot(genotype.normal, plot_path + "-normal", caption)
                    plot(genotype.reduce, plot_path + "-reduce", caption)
                    
                    writer.add_image(plot_path+'.png')
        
        else:
            logger.info("genotype = {}".format(genotype))
            
            if args.plot_path != False:
                if not os.path.isdir(os.path.join(args.plot_path, args.task_name)):
                    os.makedirs(os.path.join(args.plot_path, args.task_name))
                plot_path = os.path.join(args.plot_path, args.task_name, "EP{:02d}".format(epoch+1))
                caption = "Epoch {}".format(epoch+1)
                plot(genotype.normal, plot_path + "-normal", caption)
                plot(genotype.reduce, plot_path + "-reduce", caption)
                
                writer.add_image(plot_path+'.png')
    
    

    


    
    
    
    
    


if __name__ == '__main__':
    main()