import os
import sys
import argparse

import faiss
from data import *
from utilities import *
from networks import *
import matplotlib.pyplot as plt
import numpy as np
from domain_bus import DomainBus
from tqdm import tqdm
from centroid import *
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.optimize import linear_sum_assignment

   
def get_args():
    parser = argparse.ArgumentParser(description="Script to launch training",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #domains
    parser.add_argument("--source", help="Source" ,default='/home/tongyujun/Reserve_to_Adapt-main/data/amazon_0-9_train_all.txt')
    parser.add_argument("--target", help="Target", default='/home/tongyujun/Reserve_to_Adapt-main/data/webcam_0-9_20-30_test.txt')
    
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")   
   
    #number of classes: known, unknown and the classes of self-sup task
    parser.add_argument("--shared_classes", type=int, default=10, help="Number of classes of source domain -- known classes")
    parser.add_argument("--all_classes", type=int, default=12,help=" Known+unknown classes")
   
    #path of the folders used
    parser.add_argument("--log_dir", default="/home/tongyujun/Reserve_to_Adapt-main/log/of31/", help="Path of the log folder")
    parser.add_argument("--data_dir", default="/home/tongyujun/Office/", help="Path of the dataset")

    #to select gpu/num of workers
    parser.add_argument("--gpu", type=int, default=0, help="gpu chosen for the training")
  
    
    parser.add_argument("--use_VGG", action='store_true', default=False, help="If use VGG")
    parser.add_argument("--name", type=str, default='1')

    return parser.parse_args()


args = get_args()

orig_stdout = sys.stdout
max_iter = 10000
warmiter = 3 #a2d:2 for best result



args.log_dir = args.log_dir + args.source.split('/')[-1][0]+'2'+args.target.split('/')[-1][0]+'_'+args.name



if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
    
print('\n')    
print('TRAIN START!')
print('\n')
print('THE OUTPUT IS SAVED IN A TXT FILE HERE -------------------------------------------> ', args.log_dir)
print('\n')

f = open(args.log_dir + '/out.txt', 'w')
sys.stdout = f


def transform(data, label, is_train):

    label = one_hot(args.all_classes, label)
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        
    ])
    data = transform_train(data)
    return data, label
images,labels = get_split_dataset_info(args.source, args.data_dir)
ds = CustomDataset(images,labels,img_transformer=transform,is_train=True)
source_train = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)


def transform(data, label, is_train):
    if label in range(10):
        label = one_hot(11, label)
    else:
        label = one_hot(11,10)
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        
    ])
    data = transform_train(data)
    return data, label
images,labels = get_split_dataset_info(args.target, args.data_dir)
ds1 = CustomDataset(images,labels,img_transformer=transform,is_train=True)
target_train = torch.utils.data.DataLoader(ds1, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

def transform(data, label, is_train):
    
    label = one_hot(31,label)
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        
    ])
    data = transform_test(data)
    return data, label
ds2 = CustomDataset(images,labels,img_transformer=transform,is_train=True)
target_test = torch.utils.data.DataLoader(ds2, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)





#----------------------------load the class centroids bank
all_centroids = Centroids(class_num=args.shared_classes, dim=args.shared_classes, use_cuda=True)
discriminator = LargeAdversarialNetwork(256).cuda()
feature_extractor = ResNetFc(model_name='resnet50',model_path='/home/tongyujun/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth')
cls = CLS(feature_extractor.output_num(), args.all_classes, bottle_neck_dim=256)
net = nn.Sequential(feature_extractor, cls).cuda()


#----------------------------find virtual class
customgenearator = DomainBus([source_train, target_train])
with torch.no_grad():
    with Accumulator(['fs','ft','ls', 'lt']) as ProbRecorder:#feature_source, feature_target, label_source, label_target
        for i, ((im_source, label_source), (im_target, label_target)) in enumerate(customgenearator):
            
            im_source = im_source.cuda()
            label_source = label_source.cuda()
            im_target = im_target.cuda()
            
            _, feature_source, fc_source, predict_prob_source = net.forward(im_source)
            ft1, feature_target, fc_target, predict_prob_target = net.forward(im_target)
            fs,ft,ls,lt = [variable_to_numpy(x) for x in (feature_source, feature_target, torch.nonzero(label_source,as_tuple=True)[1], torch.nonzero(label_target,as_tuple=True)[1]) ]
            ProbRecorder.updateData(globals())
  

    s_centroids = [] #calculate source class centroids
    for i in range(args.shared_classes):
        s_centroids.append(ProbRecorder['fs'][ProbRecorder['ls']==i].mean(axis=0))
    s_centroids = np.stack(s_centroids,axis=0)

    K_cluster =20# cluster target class centroids
    faiss_kmeans = faiss.Kmeans(256, int(K_cluster), niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
    faiss_kmeans.train(ProbRecorder['ft'])
    t_centroids = faiss_kmeans.centroids
    

    #find nomatched target cluster
    cost = np.linalg.norm(s_centroids[:,None,:] -  t_centroids[None,:,:],axis=-1)
    _,t_match = linear_sum_assignment(cost)
    nomatch = []  
    for i in range(K_cluster):
        if i not in t_match:
            nomatch.append(t_centroids[i])
    nomatch = np.stack(nomatch,axis=0)
    

    fcweight = np.concatenate([s_centroids,nomatch],axis=0)
    for key, v in net.state_dict().items():   #fast initial classifier weight  
        if key=='1.main.1.2.weight':
            cost = np.linalg.norm(fcweight[:,None,:] -  v.cpu().numpy()[None,:,:],axis=-1)
            _,t_match = linear_sum_assignment(cost)
            param = torch.from_numpy(v.cpu().numpy()[t_match]).cuda().detach().clone()
            net.state_dict()['1.fc.weight'].copy_(param)  
           
    
    nomatch = torch.from_numpy(nomatch).cuda().detach().clone()
        
del(ProbRecorder)




scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=max_iter)
optimizer_discriminator = OptimWithSheduler(optim.SGD(discriminator.parameters(), lr=args.learning_rate*10, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_feature_extractor = OptimWithSheduler(optim.SGD(feature_extractor.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_cls = OptimWithSheduler(optim.SGD(cls.parameters(), lr=args.learning_rate*10, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)


##----------------------------train start
epoch = 0
k=0
best_os = 0
best_os_star = 0
best_unk = 0
best_hos = 0
best_epoch = 0
# c_weight = torch.ones(args.shared_classes)

while epoch <70:
    customgenearator = DomainBus([source_train, target_train])
    losscounter = LossCounter()
    with Accumulator(['pred_s','pred_t','label_s', 'kl','fss','ftt']) as ProbRecorder:
        
        for i, ((im_source, label_source), (im_target, label_target)) in enumerate(customgenearator):
            
            im_source = im_source.cuda()
            label_source = label_source.cuda()
            im_target = im_target.cuda()
            
            _, feature_source, fc_source, predict_prob_source = net.forward(im_source)
            ft1, feature_target, fc_target, predict_prob_target = net.forward(im_target)
            
            
            domain_prob_discriminator_1_source = discriminator.forward(feature_source)
            domain_prob_discriminator_1_target = discriminator.forward(feature_target)
            
            
            s_ctds, t_ctds = all_centroids.get_centroids()  
            _, pseudo_t_label = predict_prob_target[:,:args.shared_classes].max(1)
            

            kltarget = torch.nn.functional.kl_div( (nn.Softmax(-1)(fc_target[:,:args.shared_classes])).log(),   s_ctds[pseudo_t_label], reduction='none').sum(1).detach()
            kltarget = torch.where(torch.isinf(kltarget), torch.full_like(kltarget, 10), kltarget)

            if epoch<=1:
                gmm = GaussianMixture(n_components=3, covariance_type='full').fit(to_np(kltarget)[:,None])
            
            known_cluster = np.argmin(gmm.means_)
            unknown_cluster = np.argmax(gmm.means_)
            gmm_index = gmm.predict(to_np(kltarget)[:,None])
            
                      
            pred_s, pred_t, label_s, kl, fss, ftt \
                = [variable_to_numpy(x)  for x in (nn.Softmax(-1)(fc_source[:,:args.shared_classes]), \
                      predict_prob_target, label_source, kltarget, feature_source, feature_target)]
            ProbRecorder.updateData(globals())

            
            weight = gmm.predict_proba(to_np(kltarget)[:,None])[:,known_cluster]
            weight = torch.tensor(weight).cuda().detach()
 

            
            
            if epoch<=10:# first 10 epoch use most confident sample
                weight = torch.where(weight>0.8,torch.tensor([1]).float().cuda(),torch.tensor([0]).float().cuda()).detach()               
                r = torch.nonzero(torch.tensor(gmm_index!=known_cluster).cuda()).unsqueeze(-1)
                topk=16
                if r.size()[0]>topk:
                    r = torch.sort(kltarget.detach(),dim = 0)[1][-1*topk:]
            else:             
                weight = torch.where(torch.tensor(gmm_index==known_cluster).cuda(),torch.tensor([1]).float().cuda(),torch.tensor([0]).float().cuda()).detach()               
                r = torch.nonzero(torch.tensor(gmm_index==unknown_cluster).cuda()).unsqueeze(-1)
   

            
            feature_otherep = torch.index_select(ft1, 0, r.view(-1))
            if r.size()[0]>1:
                _, feature_otherep, logits_otherep, predict_prob_otherep = cls.forward(feature_otherep)
                _, pseudo_index = predict_prob_otherep[:,args.shared_classes:].max(1)
                pseudo_index=pseudo_index + args.shared_classes
                pseudo_label = torch.zeros(r.size()[0],args.all_classes).cuda().scatter_(1,pseudo_index.unsqueeze(1),torch.ones(r.size()[0],1).cuda())
                ce_ep = CrossEntropyLoss(pseudo_label[:,:],predict_prob_otherep[:,:])            
            else:
                ce_ep=torch.tensor(0.0)
               
            ce = CrossEntropyLoss(label_source, nn.Softmax(-1)(fc_source))

            virtual_predict_prob_source = cls.virt_forward( nomatch, feature_source, fc_source[:,:],torch.nonzero(label_source)[:,1],)
            p = torch.zeros([label_source.shape[0],nomatch.size(0)]).cuda()
            v_label_source = torch.cat((label_source[:,:],p),1)
            virtual_ce = CrossEntropyLoss(v_label_source, virtual_predict_prob_source)
    
            entropy = EntropyLoss(predict_prob_target [:,:], instance_level_weight= weight.contiguous())

            adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), predict_prob=domain_prob_discriminator_1_source )
            adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), predict_prob=1 - domain_prob_discriminator_1_target, 
                                        instance_level_weight = weight.contiguous())
               

            with OptimizerManager([optimizer_cls, optimizer_feature_extractor,optimizer_discriminator]):
                if epoch<=warmiter:
                    loss = 1 * ce + 1* virtual_ce + 0 * adv_loss + 0 * entropy + 0 * ce_ep         
                else:
                    loss = ce + 0.01 * virtual_ce + 0.3 * adv_loss + 1 * entropy + 1 * ce_ep 
                # if epoch<=warmiter: d2a
                #     loss = 1 * ce + 0* virtual_ce + 0 * adv_loss + 0 * entropy + 0 * ce_ep         
                # else:
                #     loss = ce + 0 * virtual_ce + 0.3 * adv_loss + 1 * entropy + 1 * ce_ep 
                loss.backward()
            losscounter.addOntBatch(ce, entropy, virtual_ce, ce_ep, adv_loss)
            k += 1
           
            torch.cuda.empty_cache()  

    
   
   
    all_centroids.update(ProbRecorder['pred_s'],ProbRecorder['pred_t'],ProbRecorder['label_s'])

    
    # After train
    s_centroids = []
    for i in range(args.shared_classes):
        s_centroids.append(ProbRecorder['fss'][np.nonzero(ProbRecorder['label_s'])[1]==i].mean(axis=0))
    s_centroids = np.stack(s_centroids,axis=0)

    faiss_kmeans = faiss.Kmeans(256, int(K_cluster), niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
    faiss_kmeans.train(ProbRecorder['ftt'])      
    t_centroids = faiss_kmeans.centroids

    #find nomatched target cluster
    cost = np.linalg.norm(s_centroids[:,None,:] -  t_centroids[None,:,:],axis=-1)
    _,t_match = linear_sum_assignment(cost)
    nomatch = []
    for i in range(K_cluster):
        if i not in t_match:
            nomatch.append(t_centroids[i])
    nomatch = np.stack(nomatch,axis=0)
    nomatch = torch.from_numpy(nomatch).cuda().detach().clone()

    
    if epoch ==warmiter:
        #cluster shared class+K 
        faiss_kmeans = faiss.Kmeans(256, int(args.all_classes), niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
        faiss_kmeans.train(ProbRecorder['ftt'])

        t_centroids = faiss_kmeans.centroids
        cost = np.linalg.norm(s_centroids[:,None,:] -  t_centroids[None,:,:],axis=-1)
        _,t_match = linear_sum_assignment(cost)
        #no match as unk weight
        init_unk_weight = []
        for i in range(args.all_classes):
            if i not in t_match:
                init_unk_weight.append(t_centroids[i])
        init_unk_weight = np.stack(init_unk_weight,axis=0)
        
        for key, v in net.state_dict().items():   

            if key=='1.main.1.2.weight':
                v.requires_grad = False
                net.state_dict()['1.fc.weight'].requires_grad = False
                
                vvnorm = (torch.norm(v, dim = -1)).mean().cpu().numpy()
                init_unk_weight = init_unk_weight/np.linalg.norm(init_unk_weight,axis=-1,keepdims=True)*vvnorm
                fcweight = np.concatenate([v[:args.shared_classes].clone().detach().cpu().numpy(), init_unk_weight,],axis=0)
                param = torch.from_numpy(fcweight).cuda().detach().clone()
                net.state_dict()['1.fc.weight'].copy_(param)  
                
                v.requires_grad = True
                net.state_dict()['1.fc.weight'].requires_grad = True
    
    
    
    if epoch<=30:
        gmm = BayesianGaussianMixture(n_components=4, max_iter=800).fit(ProbRecorder['kl'][:,None])
    else:
        gmm = BayesianGaussianMixture(n_components=2, max_iter=800).fit(ProbRecorder['kl'][:,None])
    torch.cuda.empty_cache()

   

    # =================================evaluation
    with TrainingModeManager([feature_extractor, cls], train=False) as mgr, Accumulator(['predict_prob','predict_index', 'label']) as accumulator:
        for (i, (im, label)) in enumerate(target_test):
            im = im.cuda()
            label = label.cuda()
            ss, fs,_,  predict_prob = net.forward(im)
            predict_prob, label = [variable_to_numpy(x) for x in (predict_prob,label)]
            label = np.argmax(label, axis=-1).reshape(-1, 1)
            predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
            accumulator.updateData(globals())
        

    for x in list(accumulator.keys()):
        globals()[x] = accumulator[x]

    y_true = label.flatten()
    y_pred = predict_index.flatten()
    m = extended_confusion_matrix(y_true, y_pred, true_labels=(list(range(args.shared_classes))+list(range(20,31))), pred_labels=list(range(args.all_classes)))

    cm = m
    cm = cm.astype(float) / np.sum(cm, axis=1, keepdims=True)
    acc_os_star = sum([cm[i][i] for i in range(args.shared_classes)]) / args.shared_classes
    unkn = sum(sum([cm[i][args.shared_classes:] for i in range(10, 21)])) / 11  
    acc_os = (acc_os_star * args.shared_classes + unkn) / 11
           
    hos = (2*acc_os_star*unkn)/(acc_os_star+unkn)                
    ce = losscounter.ce/losscounter.batch
    entropy = losscounter.entropy/losscounter.batch
    virtual = losscounter.virtual/losscounter.batch
    ce_ep = losscounter.ce_ep/losscounter.batch
    adv = losscounter.adv/losscounter.batch
    print ('Epoch:{}\tOS: {:.3f}\tOS*:{:.3f}\tUnk:{:.3f}\tHos:{:.3f}\tce: {:.3f}\tentropy:{:.3f}\tvirtual:{:.3f}\tce_ep:{:.3f}\tadv:{:.3f}'.format(epoch,acc_os,acc_os_star,unkn,hos, ce, entropy, virtual, ce_ep, adv))
    

    if hos>best_hos:
        best_os = acc_os
        best_os_star = acc_os_star
        best_unk = unkn
        best_hos = hos
        best_epoch = epoch
    torch.cuda.empty_cache()

    epoch = epoch + 1




print ('Best: Epoch:{}\tOS: {:.3f}\tOS*:{:.3f}\tUnk:{:.3f}\tHos:{:.3f}'.format(best_epoch, best_os,best_os_star,best_unk,best_hos))
print('class_num'+ str(args.all_classes)   + str(args))
sys.stdout = orig_stdout
f.close()

