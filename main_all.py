import faiss
import faiss.contrib.torch_utils
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import numpy
import time
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, ADE20KDataset, LvisDataset, CocoStuff10k
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import pickle
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from PIL import Image
import matplotlib as plt
from tensorboardX import SummaryWriter
import gc
print(torch.__version__)

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['lvis','voc', 'cityscapes','ade20k','coco'], help='Name of dataset')
    parser.add_argument("--num_channels", type=int, default=6,
                        help="num channels in last layer (default: None)")
    parser.add_argument("--num_neighbours", type=int, default=6,
                        help="num of neighbours for softmax (default: None)")
    parser.add_argument("--num_classes", type=int, default=21,
                        help="num classes in the dataset (default: None)")
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3plus_resnet50',
                                 'deeplabv3plus_resnet101',
                                 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--reduce_dim", action='store_true',default=False)
    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--freeze_backbone", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=200e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--temp", type=float, default=20.0,
                        help="multiplying factor for norm loss")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step', 'multi_poly'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='nn_cross_entropy',
                        choices=['cross_entropy','nn_cross_entropy'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')
    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints',
                        help='directory to save checkpoints')
    parser.add_argument("--vis_num_samples", type=int, default=15,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    elif opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)

    elif opts.dataset == 'ade20k':
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        if opts.dataset == 'ade20k':
            train_dst = ADE20KDataset(ROOT_DIR=opts.data_root, 
                                      period='train', transform=train_transform)
            val_dst = ADE20KDataset(ROOT_DIR=opts.data_root,
                                      period='val', transform=val_transform)
    elif opts.dataset == 'lvis' :
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(target_type='int16'),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(target_type='int16'),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(target_type='int16'),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = LvisDataset(ROOT_DIR=opts.data_root, 
                                      period='train', transform=train_transform)
        val_dst = LvisDataset(ROOT_DIR=opts.data_root, 
                                      period='val', transform=val_transform)
    elif opts.dataset == 'coco':
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = CocoStuff10k(root=opts.data_root,split='train', 
                                  val=False)
        val_dst = CocoStuff10k(root=opts.data_root,split='test',
                                  val=True)
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics,ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0
    if(opts.reduce_dim):
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        gpu_index = faiss.GpuIndexFlatL2(res,int(opts.num_channels))

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            if(opts.dataset in ['coco','voc','cityscapes']):
                labels[labels==255]  = -1
            if(opts.dataset == 'ade20k' or opts.dataset == 'lvis'):
                labels = labels - 1
            if(opts.reduce_dim):
                outputs , class_emb = model(images)
                if(i==0):
                    gpu_index.add(class_emb)
                trans_outputs = torch.transpose(torch.transpose(outputs,1,2),2,3).reshape(outputs.size()[0]*outputs.size()[2]*outputs.size()[3],opts.num_channels)
                trans_outputs = trans_outputs.contiguous()
                D,I = gpu_index.search(trans_outputs,1)
                preds = torch.transpose(torch.transpose(I.reshape(outputs.size()[0],outputs.size()[2],outputs.size()[3],1),2,3),1,2)
                preds = preds.squeeze(1).detach().cpu().numpy()
            else:
                outputs = model(images)
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)

            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]
                    if(opts.dataset.lower()=='coco'):
                        image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                        target = loader.dataset._colorize_mask(target).convert('RGB')
                        pred = loader.dataset._colorize_mask(pred).convert('RGB')
                        Image.fromarray(image).save('results/%d_image.png' % img_id)
                        target.save('results/%d_target.png' % img_id)
                        pred.save('results/%d_pred.png' % img_id)
                    else:
                        image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                        target = loader.dataset.decode_target(target).astype(np.uint8)
                        pred = loader.dataset.decode_target(pred).astype(np.uint8)
                        Image.fromarray(image).save('results/%d_image.png' % loader.dataset.img_ids[i])
                        Image.fromarray(target).save('results/%d_target.png' % loader.dataset.img_ids[i])
                        Image.fromarray(pred).save('results/%d_pred.png' % loader.dataset.img_ids[i])
                    img_id += 1
        torch.cuda.empty_cache()
        del  images, labels, outputs,  preds
        if(opts.reduce_dim):
            del I, D, trans_outputs, class_emb, gpu_index
        gc.collect()
        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        ignore_index = 255
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        ignore_index = 255
    elif opts.dataset.lower() =='ade20k':
        opts.num_classes = 150
        ignore_index = -1
    elif opts.dataset.lower() =='lvis':
        opts.num_classes = 1284 
        ignore_index = -1
    elif opts.dataset.lower() == 'coco':
        opts.num_classes = 182
        ignore_index = 255
    if(opts.reduce_dim==False):
        opts.num_channels = opts.num_classes
    if(opts.test_only==False):
        writer = SummaryWriter('summary/'+opts.vis_env)
    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1
    
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))
    epoch_interval = int(len(train_dst)/opts.batch_size)
    if(epoch_interval>5000):
        opts.val_interval = 5000
    else:
        opts.val_interval = epoch_interval
    print("Evaluation after %d iterations" % (opts.val_interval))

    # Set up model
    model_map = {
        #'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        #'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        #'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }
    if(opts.reduce_dim):
        num_classes_input = [opts.num_channels,opts.num_classes]
    else:
        num_classes_input = [opts.num_classes]
    model = model_map[opts.model](num_classes=num_classes_input, output_stride=opts.output_stride, reduce_dim= opts.reduce_dim)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)
    if opts.reduce_dim:
        emb_layer = ['embedding.weight']
        params_classifier = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in emb_layer, model.classifier.named_parameters()))))
        params_embedding = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in emb_layer, model.classifier.named_parameters()))))
        if opts.freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False
            optimizer = torch.optim.SGD(params=[
                #@{'params': model.backbone.parameters(),'lr':0.1*opts.lr},
                {'params': params_classifier, 'lr': opts.lr},
                {'params':params_embedding,'lr':opts.lr,'momentum':0.95},
                ], lr=opts.lr, momentum=0.9,weight_decay=opts.weight_decay)
        else:
            optimizer = torch.optim.SGD(params=[
                {'params': model.backbone.parameters(),'lr':0.1*opts.lr},
                {'params': params_classifier, 'lr': opts.lr},
                {'params':params_embedding,'lr':opts.lr},
                ], lr=opts.lr, momentum=0.9,weight_decay=opts.weight_decay)
    # Set up optimizer
    else:
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    elif opts.lr_policy == 'multi_poly':
        scheduler = utils.MultiPolyLR(optimizer, opts.total_itrs, power =[0.9,0.9,0.95])

    # Set up criterion
    if(opts.reduce_dim):
        opts.loss_type = 'nn_cross_entropy'
    else:
        opts.loss_type = 'cross_entropy'

    if opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    elif opts.loss_type == 'nn_cross_entropy':
        criterion = utils.NNCrossEntropy(ignore_index=ignore_index, reduction = 'mean', num_neighbours = opts.num_neighbours, temp= opts.temp, dataset=opts.dataset)


    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir(opts.checkpoint_dir)
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        increase_iters =True
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("scheduler state dict :",scheduler.state_dict())
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0

    writer.add_text('lr',str(opts.lr))
    writer.add_text('batch_size',str(opts.batch_size))
    writer.add_text('reduce_dim',str(opts.reduce_dim))
    writer.add_text('checkpoint_dir',opts.checkpoint_dir)
    writer.add_text('dataset',opts.dataset)
    writer.add_text('num_channels',str(opts.num_channels))
    writer.add_text('num_neighbours',str(opts.num_neighbours))
    writer.add_text('loss_type',opts.loss_type)
    writer.add_text('lr_policy',opts.lr_policy)
    writer.add_text('temp',str(opts.temp))
    writer.add_text('crop_size',str(opts.crop_size))
    writer.add_text('model',opts.model)
    accumulation_steps = 1
    writer.add_text('accumulation_steps',str(accumulation_steps))
    j = 0
    updateflag = False
    while True: 
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            if(opts.dataset == 'ade20k' or opts.dataset=='lvis'):
                labels = labels-1

            optimizer.zero_grad()
            if(opts.reduce_dim):
                outputs, class_emb = model(images)
                loss = criterion(outputs, labels, class_emb)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            j = j + 1
            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)
                vis.vis_scalar('LR',cur_itrs,scheduler.state_dict()['_last_lr'][0])
            torch.cuda.empty_cache()
            del images, labels, outputs, loss
            if(opts.reduce_dim):
                del class_emb
            gc.collect()
            if (cur_itrs) % 50 == 0 :
                interval_loss = interval_loss/50
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                writer.add_scalar('Loss',interval_loss,cur_itrs)
                writer.add_scalar('lr',scheduler.state_dict()['_last_lr'][0],cur_itrs)
            if cur_itrs % opts.val_interval == 0:
                save_ckpt(opts.checkpoint_dir + '/latest_%d.pth' % (cur_itrs))
            if cur_itrs % opts.val_interval == 0:
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(opts.checkpoint_dir + '/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset,opts.output_stride))

                writer.add_scalar('[Val] Overall Acc',val_score['Overall Acc'],cur_itrs)
                writer.add_scalar('[Val] Mean IoU',val_score['Mean IoU'],cur_itrs)
                writer.add_scalar('[Val] Mean Acc',val_score['Mean Acc'],cur_itrs)
                writer.add_scalar('[Val] Freq Acc',val_score['FreqW Acc'],cur_itrs)


                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        if(opts.dataset.lower()=='coco'):
                            target = numpy.asarray(train_dst._colorize_mask(target).convert('RGB')).transpose(2, 0, 1).astype(np.uint8)
                            lbl = numpy.asarray(train_dst._colorize_mask(lbl).convert('RGB')).transpose(2, 0, 1).astype(np.uint8)
                        else:
                            target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                            lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()  
            if cur_itrs >=  opts.total_itrs:
                return
    writer.close()
        
if __name__ == '__main__':
    main()
