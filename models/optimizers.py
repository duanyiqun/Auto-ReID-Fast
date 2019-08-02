
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def build_base_SGD_opt(args, model):

    if not args.PCB:
        ignored_params = list(map(id, model.classifier.parameters() ))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
                {'params': base_params, 'lr': 0.1*args.backbone_lr},
                {'params': model.classifier.parameters(), 'lr': args.head_lr}
            ], weight_decay=args.weight_decay, momentum=args.momentum, nesterov=args.nesterov)
    else:
        ignored_params = list(map(id, model.model.fc.parameters() ))
        ignored_params += (list(map(id, model.classifier0.parameters() )) 
                        +list(map(id, model.classifier1.parameters() ))
                        +list(map(id, model.classifier2.parameters() ))
                        +list(map(id, model.classifier3.parameters() ))
                        +list(map(id, model.classifier4.parameters() ))
                        +list(map(id, model.classifier5.parameters() ))
                        #+list(map(id, model.classifier6.parameters() ))
                        #+list(map(id, model.classifier7.parameters() ))
                        )
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
                {'params': base_params, 'lr': 0.1*args.backbone_lr},
                {'params': model.model.fc.parameters(), 'lr': args.head_lr},
                {'params': model.classifier0.parameters(), 'lr': args.head_lr},
                {'params': model.classifier1.parameters(), 'lr': args.head_lr},
                {'params': model.classifier2.parameters(), 'lr': args.head_lr},
                {'params': model.classifier3.parameters(), 'lr': args.head_lr},
                {'params': model.classifier4.parameters(), 'lr': args.head_lr},
                {'params': model.classifier5.parameters(), 'lr': args.head_lr},
                #{'params': model.classifier6.parameters(), 'lr': 0.01},
                #{'params': model.classifier7.parameters(), 'lr': 0.01}
            ], weight_decay=args.weight_decay, momentum=args.momentum, nesterov=args.nesterov)
    
    return optimizer_ft

def wrap_exp_lr(args, optimizer):
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=0.1)
    return exp_lr_scheduler

def wrap_cosine_lr(args, optimizer):
    cosine_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.w_lr_min)
    return cosine_lr_scheduler

def get_lr_scheduler(args, optimizer):
    return eval(args.lr_mode)(args, optimizer)
    

def get_optimizer(args, model):
    return eval(args.optimizer_type)(args, model)