#########
# imports
#########
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import umap
import torch
from sklearn.cluster import KMeans
import h5py
import seaborn as sb
import pandas as pd
from sklearn.decomposition import PCA
import random

from ava.data.data_container import DataContainer
from ava.data.data_container_go import DataContainerGo
from ava.models.vae import VAE
from ava.models.go_vae import goVAE
from torch.distributions import LowRankMultivariateNormal
from ava.plotting.mmd_plots import _calculate_mmd


from ava.models.vae_dataset import get_syllable_partition, get_syllable_data_loaders
from ava.plotting.latent_projection import latent_projection_plot_DC
from ava.plotting.grid_plot import grid_plot
from ava.models.vae_dataset import get_syllable_partition, get_syllable_data_loaders, get_hdf5s_from_dir

######
# steps:
# 1. get bird 1 model, bird 2 model
# 2. get bird 1 centroids, bird 2 centroids
# 3. get bird 2 centroids through bird 1 model
# 4. look at distortion/map between bird 1, bird 2 models
def get_point_prob(points,old_model,new_model):

#expects as input two trained VAEs (with loaded state), set of points to compare
    log_prob_new_encode_old = 0.0
    log_prob_old_encode_new = 0.0
    log_prob_old_encode_old = 0.0
    log_prob_new_encode_new = 0.0
    for ind in range(points.shape[0]):
        #print(points.shape)
        #print(points[ind,:,:].shape)
        with torch.no_grad():
            o_mu,o_u,o_d = old_model.encode(points[ind,:,:].unsqueeze(dim=0))
            n_mu,n_u,n_d = new_model.encode(points[ind,:,:].unsqueeze(dim=0))

        orig_distrib = LowRankMultivariateNormal(o_mu,o_u,o_d)
        new_distrib = LowRankMultivariateNormal(n_mu,n_u,n_d)

        #o_mu = o_mu.detach().cpu().numpy()
        #n_mu = n_mu.detach().cpu().numpy()

        #print('Old projection into old space probability:' + str(orig_distrib.log_prob(o_mu).detach().cpu().numpy()))
        #print('New Projection into new space probability:' + str(new_distrib.log_prob(n_mu).detach().cpu().numpy()))
        #print('Old - new probability at mean:' + str((orig_distrib.log_prob(o_mu) - new_distrib.log_prob(n_mu)).detach().cpu().numpy()))
        log_prob_old_encode_new += orig_distrib.log_prob(n_mu)
        log_prob_new_encode_old += new_distrib.log_prob(o_mu)
        log_prob_old_encode_old += orig_distrib.log_prob(o_mu)
        log_prob_new_encode_new += new_distrib.log_prob(n_mu)

    log_old_distrib_prob_ratio = log_prob_old_encode_old - log_prob_old_encode_new
    log_new_distrib_prob_ratio = log_prob_new_encode_old - log_prob_new_encode_new

    return log_prob_new_encode_old, log_prob_old_encode_new, log_old_distrib_prob_ratio, log_new_distrib_prob_ratio



root = '/home/mmartinez/autoencoded-vocal-analysis/ava_test3'

beta_list = list(np.arange(0.5,3.6,0.75))
print(beta_list)
mod_train_beta = 2.75
trainDays = ['100']

train_new = ['blu288']
train_orig = ['blu258']
plots_dir = os.path.join(root,'go_plots_betas_compare_new')
if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)

model_filename_old = os.path.join(root,'model_258_0', 'checkpoint_100.tar')
#model_filename_go = os.path.join(root, 'model_258_1', 'checkpoint_100.tar')
model_filename_betas = [os.path.join(root,'model_288_beta_' + str(beta),'checkpoint_100.tar') for beta in beta_list]

model_beta_new = os.path.join(root,'model_288_2_beta_' + str(mod_train_beta),'checkpoint_100.tar')
#model_filename_both_loss = os.path.join(root, 'model_288_both_loss', 'checkpoint_100.tar')
#model_filename_old_loss = os.path.join(root, 'model_288_old_loss', 'checkpoint_100.tar')
#model_filename_new_loss = os.path.join(root, 'model_288_new_loss', 'checkpoint_100.tar')

audio_old = [os.path.join(root,animal,'audio',day) for animal in train_orig for day in trainDays]
segment_old = [os.path.join(root,animal,'segs',day) for animal in train_orig for day in trainDays]
spec_old = [os.path.join(root,animal,'h5s',day) for animal in train_orig for day in trainDays]
proj_old = [os.path.join(root,animal,'proj',day) for animal in train_orig for day in trainDays]

audio_new = [os.path.join(root,animal,'audio',day) for animal in train_new for day in trainDays]
segment_new = [os.path.join(root,animal,'segs',day) for animal in train_new for day in trainDays]
spec_new = [os.path.join(root,animal,'h5s',day) for animal in train_new for day in trainDays]
proj_new = [os.path.join(root,animal,'proj',day) for animal in train_new for day in trainDays]

#old_data_old_model =  DataContainer(projection_dirs=proj_old, audio_dirs=audio_old, \
#    segment_dirs=segment_old,plots_dir=plots_dir, \
#	spec_dirs=spec_old, model_filename=model_filename_old)

#old_data_new_model =  DataContainer(projection_dirs=proj_old, audio_dirs=audio_old, \
#    segment_dirs=segment_old,plots_dir=plots_dir, \
#	spec_dirs=spec_old, model_filename=model_filename_go)

#new_data_old_model =  DataContainer(projection_dirs=proj_new, audio_dirs=audio_new, \
#    segment_dirs=segment_new,plots_dir=plots_dir, \
#	spec_dirs=spec_new, model_filename=model_filename_old)

#new_data_new_model =  DataContainer(projection_dirs=proj_new, audio_dirs=audio_new, \
#    segment_dirs=segment_new,plots_dir=plots_dir, \
#	spec_dirs=spec_new, model_filename=model_filename_go)

spec_centroids = [os.path.join(root,animal,'old_mod_spec_centroids_v3',day) for animal in train_orig for day in trainDays]
save1_dir = os.path.join(root,'model_258_0')
save2_dir = os.path.join(root,'model_258_1')
save_beta_list = [os.path.join(root,'model_288_beta_' + str(beta)) for beta in beta_list]
save_beta_new = os.path.join(root,'model_288_2_beta_' + str(mod_train_beta))
#save_both_loss = os.path.join(root,'model_288_both_loss')
#save_old_loss = os.path.join(root,'model_288_old_loss')
#save_new_loss = os.path.join(root,'model_288_old_loss')

filenames=[]

for dir in spec_centroids:
    filenames += get_hdf5s_from_dir(dir)

#spec_centroid_tensors = []

for fname in filenames:
    with h5py.File(fname, 'r') as f:
        spec = f['specs'][:]

    spec = torch.from_numpy(spec).type(torch.FloatTensor)
#    print(spec.shape)
    spec_centroid_tensors = spec

#print(model_filename_go)
'''
new_data_fnames = get_hdf5s_from_dir(spec_new[0])
old_data_fnames = get_hdf5s_from_dir(spec_old[0])

new_data = []
old_data = []

for ind in range(len(new_data_fnames)):
    n_f= new_data_fnames[ind]

    with h5py.File(n_f, 'r') as f:
        spec = f['specs'][:]
        spec = torch.from_numpy(spec).type(torch.FloatTensor)

    new_data.append(spec)

for ind in range(len(old_data_fnames)):
    o_f = old_data_fnames[ind]
    with h5py.File(o_f, 'r') as f:
        spec = f['specs'][:]
        spec = torch.from_numpy(spec).type(torch.FloatTensor)

    old_data.append(spec)

old_data = torch.stack(old_data).view(-1,128,128)
new_data = torch.stack(new_data).view(-1,128,128)
'''

orig_model = VAE(save_dir=save1_dir)
orig_model.load_state(model_filename_old)
#new_model = goVAE(save_dir=save2_dir,oldVAE=orig_model)
#new_model = goVAE(save_dir=save2_dir)
#new_model.load_state(model_filename_go)

model_betas = []
for ind, fname in enumerate(model_filename_betas):
    beta = beta_list[ind]
    model_beta_tmp = goVAE(save_dir=save_beta_list[ind],use_old=1,use_new=1,\
                           beta=beta)
    model_beta_tmp.load_state(fname)
    model_betas.append(model_beta_tmp)
    #print(ind + 2)
model_beta_tmp = goVAE(save_dir=save_beta_new,use_old=1,use_new=1,\
                            beta=2.75,zhat=True)

model_beta_tmp.load_state(model_beta_new)
model_betas.append(model_beta_tmp)
#print(len(model_betas))


#new_model_both_loss = goVAE(save_dir=save_both_loss,use_old=1,use_new=1)
#new_model_both_loss.load_state(model_filename_both_loss)
#new_model_old_loss = goVAE(save_dir=save_old_loss,use_old=1,use_new=0)
#new_model_old_loss.load_state(model_filename_old_loss)
#new_model_new_loss = goVAE(save_dir=save_new_loss,use_old=0,use_new=1)
#new_model_new_loss.load_state(model_filename_new_loss)


#op_log_prob, np_log_prob, old_prob_ratio,new_prob_ratio = get_point_prob(spec_centroid_tensors,orig_model,new_model)
#print('Log probability of old encoding points in new distribution:' + str(op_log_prob))
#print('Log probability of new encoding points in old distribution:' + str(np_log_prob))
#print('Log probability ratio of old encoding points:' + str(old_prob_ratio))
#print('Log probability ratio of new encoding points:' + str(new_prob_ratio))
#print(len(spec_centroid_tensors))
new_encodings = []
with torch.no_grad():
    orig_nearest_points,_,_ = orig_model.encode(spec_centroid_tensors)
    #encoded_nearest_points,_,_ = new_model.encode(spec_centroid_tensors)
    for ind, model in enumerate(model_betas):
        tmp_mu,_,_ = model.encode(spec_centroid_tensors)
        tmp_mu = tmp_mu.cpu().detach().numpy()
        new_encodings.append(tmp_mu)
#        print(ind + 2)
    #encoded_both_loss,_,_ = new_model_both_loss.encode(spec_centroid_tensors)
    #encoded_old_loss,_,_ = new_model_old_loss.encode(spec_centroid_tensors)
    #encoded_new_loss,_,_ = new_model_new_loss.encode(spec_centroid_tensors)
#print(len(new_encodings))
#print(orig_nearest_points)
#print(encoded_nearest_points)
#print(orig_nearest_points.shape)
#print(encoded_both_loss.shape)
orig_nearest_points = orig_nearest_points.cpu().detach().numpy()
#encoded_nearest_points = encoded_nearest_points.cpu().detach().numpy()
#encoded_both_loss = encoded_both_loss.cpu().detach().numpy()
#encoded_old_loss = encoded_old_loss.cpu().detach().numpy()
#encoded_new_loss = encoded_new_loss.cpu().detach().numpy()
z_dim = orig_nearest_points.shape[1]
#orig_dists = np.linalg.norm(orig_nearest_points - encoded_nearest_points,axis=1)

loss_comp_latents = [orig_nearest_points] + new_encodings



#all_orig_latents = np.vstack((orig_nearest_points,encoded_nearest_points))
'''
new_model_new_latents = new_data_new_model.request('latent_means')
old_model_new_latents = new_data_old_model.request('latent_means')
new_model_old_latents = old_data_new_model.request('latent_means')
old_model_old_latents = old_data_old_model.request('latent_means')
'''
print('Getting new_latents')
num_workers = min(7,os.cpu_count()-1)
split = 0.25
old_data_partition = get_syllable_partition(spec_old,split,shuffle=False)
new_data_partition = get_syllable_partition(spec_new,split,shuffle=False)


old_data_loader = get_syllable_data_loaders(old_data_partition, batch_size=128,num_workers=num_workers)
new_data_loader = get_syllable_data_loaders(new_data_partition, batch_size=128,num_workers=num_workers)

#old_data_loader['test'] = old_data_loader['train']
#new_data_loader['test'] = new_data_loader['train']
test_loss = []
train_loss = []

print('='*40)
print('New data loss on original model')
_,test_loss_orig,orig_rec_new = orig_model.test_epoch(new_data_loader['test'])
print('='*40)
print('Old data loss on original model')
_,tr_loss, orig_rec_old = orig_model.test_epoch(old_data_loader['test'])

test_loss.append(test_loss_orig)
train_loss.append(tr_loss)

#print('Test loss on both loss model')
#test_loss_both = new_model_both_loss.test_epoch(new_data_loader['test'])
#print('='*40)

#print('Test loss on old loss model')
#test_loss_old = new_model_old_loss.test_epoch(new_data_loader['test'])
#print('='*40)

#new_model_new_latents = new_model.get_latent(new_data_loader['train'])
#old_model_new_latents = orig_model.get_latent(new_data_loader['train'])
#print('Getting orig latents')

new_data_latents = []
old_data_latents = []

old_data_inds = np.ones((len(orig_nearest_points),))
orig_model_old_latents = orig_model.get_latent(old_data_loader['test'])
orig_model_new_latents = orig_model.get_latent(new_data_loader['test'])

#orig_rec_old = orig_model.decode(torch.from_numpy(orig_model_old_latents).type(torch.FloatTensor))
#orig_rec_new = orig_model.decode(torch.from_numpy(orig_model_new_latents).type(torch.FloatTensor))

embedding_ind_list = [old_data_inds]

new_data_inds = [np.ones((len(orig_model_new_latents)))]
old_data_inds = [np.ones((len(orig_model_old_latents)))]
new_data_latents = [orig_model_new_latents]
old_data_latents = [orig_model_old_latents]

new_data_recs = [orig_rec_new]
old_data_recs = [orig_rec_old]

# get_latent orders specs in the same way
beta_list.append(mod_train_beta)

#print(model_betas)
for b_ind, model in enumerate(model_betas):

    print('='*40)
    print('Getting latents for new model ' + str(b_ind + 1))
    print('New data loss with beta =' + str(beta_list[b_ind]))
    _,t_loss,new_rec = model.test_epoch(new_data_loader['test'])
    train_loss.append(t_loss)
    print('='*40)
    print('Old data loss with beta =' + str(beta_list[b_ind]))
    _,tr_loss,old_rec = model.test_epoch(old_data_loader['test'])
    test_loss.append(tr_loss)
    print('='*40)

    new_tmp = model.get_latent(new_data_loader['test'])
    old_tmp = model.get_latent(old_data_loader['test'])
    new_data_latents.append(new_tmp)
    old_data_latents.append(old_tmp)

    new_data_recs.append(new_rec)
    old_data_recs.append(old_rec)


    embedding_ind_list.append((b_ind + 2)*np.ones(len(new_encodings[b_ind])))
    old_data_inds.append((b_ind + 2)*np.ones(len(old_tmp)))
    new_data_inds.append((b_ind + 2)*np.ones(len(new_tmp)))
    #print(len(new))

print(tr_loss)
print(len(test_loss))
print(len(train_loss))

print(len(embedding_ind_list))
print(np.hstack(embedding_ind_list).shape)

ax = plt.gca()
x_ax= list(range(len(test_loss)))
print(len(x_ax))
print(len(test_loss[1:-1]))
#plt.plot(x_ax[0],train_loss[0], 'orange')
#plt.plot(x_ax[0],test_loss[0], 'blue')
bird1, = plt.plot(x_ax,train_loss, 'orange')
bird2, = plt.plot(x_ax,test_loss, 'blue')

plt.xticks(list(range(len(train_loss))), ['Orig. Model', 'Beta = 0.5', 'Beta = 1.25', 'Beta = 2.0', 'Beta = 2.75', 'Beta = 3.5','Beta new train type'])
plt.legend([bird1, bird2], ['Train Set', 'Test Set'])
save_filename=os.path.join(plots_dir,'rec_err_plot_betas.pdf')

plt.ylabel('Reconstruction Error')
plt.tight_layout()
plt.savefig(save_filename)
plt.close('all')


#Latent Differences (MSE)

latent_difs_new = []
latent_difs_old = []
for ind, latents in enumerate(new_data_latents):
    latent_difs_new.append(np.sum((latents - orig_model_new_latents)**2,axis=1))
    latent_difs_old.append(np.sum((old_data_latents[ind] - orig_model_old_latents)**2,axis=1))

rec_difs_new = []
rec_difs_old = []

for ind, recs in enumerate(new_data_recs):
    rec_difs_new.append(np.sum((recs.reshape((recs.shape[0],-1)) - orig_rec_new.reshape((orig_rec_new.shape[0],-1)))**2,axis=1))
    rec_difs_old.append(np.sum((old_data_recs[ind].reshape((old_data_recs[ind].shape[0],-1)) \
                                - orig_rec_old.reshape((orig_rec_old.shape[0],-1)))**2,axis=1))
######
# MSE Plot
#########
embed_inds = np.hstack(embedding_ind_list)

latents = np.vstack(latent_difs_old[1:-1])
#latents = np.vstack(loss_comp_latents)
sel_ol = random.sample(range(len(old_data_recs[ind])), len(old_data_recs[ind])//10)
sel_n = random.sample(range(len(recs)), len(recs)//10)
'''
mse_b_05_orig = np.sum((latents[embed_inds == 1,:] - latents[embed_inds == 2,:])**2,axis=1)/z_dim
mse_b_125_orig = np.sum((latents[embed_inds == 1,:] - latents[embed_inds == 3,:])**2,axis=1)/z_dim
mse_b_2_orig = np.sum((latents[embed_inds == 1,:] - latents[embed_inds == 4,:])**2,axis=1)/z_dim
mse_b_275_orig = np.sum((latents[embed_inds == 1,:] - latents[embed_inds == 5,:])**2,axis=1)/z_dim
mse_b_35_orig = np.sum((latents[embed_inds == 1,:] - latents[embed_inds == 6,:])**2,axis=1)/z_dim
'''
mse_dict = {'mse_beta05':rec_difs_old[1][sel_ol], 'mse_beta125':rec_difs_old[2][sel_ol], 'mse_beta_2':rec_difs_old[3][sel_ol],\
            'mse_beta275':rec_difs_old[4][sel_ol], 'mse_beta35':rec_difs_old[5][sel_ol], 'mse_new_loss':rec_difs_old[6][sel_ol]}
mse_dat = pd.DataFrame(data=mse_dict,dtype=np.float64)
#mse_dat["id"] = mse_dat.index
#mse_dat = pd.wide_to_long(mse_dat,'mse',i='id',j='mse')
sb.set_theme(style="whitegrid")

#ax = sb.boxplot(x=[''])
save_filename=os.path.join(plots_dir,'mseplot_recs_old.pdf')
ax = sb.swarmplot(data = mse_dat,size=0.8)
fig = ax.get_figure()
fig.savefig(save_filename)
plt.close('all')


mse_dict = {'mse_beta05':rec_difs_new[1][sel_n], 'mse_beta125':rec_difs_new[2][sel_n], 'mse_beta_2':rec_difs_new[3][sel_n],\
            'mse_beta275':rec_difs_new[4][sel_n], 'mse_beta35':rec_difs_new[5][sel_n], 'mse_new_loss':rec_difs_new[6][sel_n]}
mse_dat = pd.DataFrame(data=mse_dict,dtype=np.float64)
#mse_dat["id"] = mse_dat.index
#mse_dat = pd.wide_to_long(mse_dat,'mse',i='id',j='mse')
sb.set_theme(style="whitegrid")

#ax = sb.boxplot(x=[''])
save_filename=os.path.join(plots_dir,'mseplot_recs_new.pdf')
ax = sb.swarmplot(data = mse_dat,size=0.8)
fig = ax.get_figure()
fig.savefig(save_filename)
plt.close('all')
######
# Plotting location of all key points with each beta value.

##########

print("Running UMAP... (n="+str(len(np.vstack(loss_comp_latents)))+")")
transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
    metric='euclidean', random_state=42)

embed = transform.fit_transform((np.vstack(loss_comp_latents)))

embed_inds = np.hstack(embedding_ind_list)
print('Done!')
print('UMAP Plot')
ax = plt.gca()

## plot 1 ###############################

#for ind in range(len(both_ind)):
#    ax.plot([old_embed[ind,0],orig_embed[ind,0]],\
#            [old_embed[ind,1],orig_embed[ind,1]], '--k', linewidth=0.5)
orig_dat = embed[embed_inds == 1,:]
b_05_dat = embed[embed_inds == 2,:]
b_125_dat = embed[embed_inds == 3,:]
b_20_dat = embed[embed_inds == 4,:]
b_275_dat = embed[embed_inds == 5,:]
b_35_dat = embed[embed_inds == 6,:]
b_275_2_dat = embed[embed_inds == 7,:]
orig_embed = ax.scatter(orig_dat[:,0],orig_dat[:,1],color='k',marker='o')
b_05 = ax.scatter(b_05_dat[:,0],b_05_dat[:,1],color='b',marker='.')
b_125 = ax.scatter(b_125_dat[:,0],b_125_dat[:,1],color='r',marker='.')
b_20 = ax.scatter(b_20_dat[:,0],b_20_dat[:,1],color='g',marker='.')
b_275 = ax.scatter(b_275_dat[:,0],b_275_dat[:,1],color='c',marker='.')
b_35 = ax.scatter(b_35_dat[:,0],b_35_dat[:,1],color='m',marker='.')
b_275_2 = ax.scatter(b_275_2_dat[:,0],b_275_2_dat[:,1],color='y',marker='.')
#b_05 = ax.scatter()

#im5 = ax.scatter(embed[0:24,0],embed[0:24,1],color = 'blue')
#im6 = ax.scatter(embed[25::,0], embed[25::,1], color= 'red')
#plt.scatter(embed[0:24,0],embed[0:24,1],'o')
#plt.scatter(embed[25::,0],embed[25::,1],'x')
save_filename=os.path.join(plots_dir,'umap_keypoints_betas.pdf')
#print("total number of key points plotted: "+ str(np.shape(loss_comp_latents)[0]))
plt.legend([orig_embed,b_05,b_125,b_20,b_275,b_35,b_275_2],['orig embed','b=0.5','b=1.25','b=2.0','b=2.75','b=3.5', 'b=2.75 new'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

##############
# plotting original, b = 2.75 case
#################################

ax = plt.gca()
ol_inds = np.hstack([old_data_inds[0],old_data_inds[-1]])
ol_data = np.vstack([old_data_latents[0],old_data_latents[-1]])
ol_inds[ol_inds > 1] = 2
nl_inds = np.hstack([new_data_inds[0],new_data_inds[-1]])
nl_inds[nl_inds > 1] = 4
nl_inds[nl_inds == 1] = 3
nl_data = np.vstack([new_data_latents[0],new_data_latents[-1]])
all_inds = np.hstack([ol_inds, nl_inds])
print(np.unique(all_inds))

embed = transform.fit_transform(np.vstack([ol_data,nl_data]))
orig_dat = embed[all_inds == 1,:]
beta_275_old = embed[all_inds == 2,:]
orig_new_dat = embed[all_inds == 3,:]
beta_275_new = embed[all_inds == 4.,:]

orig_embed_old = ax.scatter(orig_dat[0,0],orig_dat[0,1],color='k',marker='o',)
beta_35_od = ax.scatter(beta_35_old[0,0],beta_35_old[0,1],color='b',marker='.')
orig_embed_new = ax.scatter(orig_new_dat[0,0],orig_new_dat[0,1],color='r',marker='.')
beta_35_nd = ax.scatter(beta_35_new[0,0],beta_35_new[0,1],color='g',marker='.')

ax.scatter(orig_dat[:,0],orig_dat[:,1],color='k',marker='o',s=0.1,alpha=0.1)
ax.scatter(beta_275_old[:,0],beta_275_old[:,1],color='b',marker='.',s=0.1,alpha=0.1)
ax.scatter(orig_new_dat[:,0],orig_new_dat[:,1],color='r',marker='.',s=0.1,alpha=0.1)
ax.scatter(beta_275_new[:,0],beta_275_new[:,1],color='g',marker='.',s=0.1,alpha=0.1)



save_filename=os.path.join(plots_dir,'umap_orig_beta35_comp.pdf')
#print("total number of key points plotted: "+ str(np.shape(loss_comp_latents)[0]))
plt.legend([orig_embed_old,orig_embed_new,beta_35_od,beta_35_nd],['orig embed old data','orig embed new data','b=2.75 old data','b=2.75.5 new data'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

'''
##############
#Comparisons between old data and all other model types
# first: 2.75
ax = plt.gca()
ol_inds = np.hstack([old_data_inds[0],old_data_inds[-2]])
ol_data = np.vstack([old_data_latents[0],old_data_latents[-2]])
ol_inds[ol_inds > 1] = 2
nl_inds = np.hstack([new_data_inds[0],new_data_inds[-2]])
nl_inds[nl_inds > 1] = 4
nl_inds[nl_inds == 1] = 3
nl_data = np.vstack([new_data_latents[0],new_data_latents[-2]])
all_inds = np.hstack([ol_inds, nl_inds])
print(np.unique(all_inds))

embed = transform.fit_transform(np.vstack([ol_data,nl_data]))
orig_dat = embed[all_inds == 1,:]
beta_275_old = embed[all_inds == 2,:]
orig_new_dat = embed[all_inds == 3,:]
beta_275_new = embed[all_inds == 4.,:]

orig_embed_old = ax.scatter(orig_dat[0,0],orig_dat[0,1],color='k',marker='o',)
beta_275_od = ax.scatter(beta_275_old[0,0],beta_275_old[0,1],color='b',marker='.')
orig_embed_new = ax.scatter(orig_new_dat[0,0],orig_new_dat[0,1],color='r',marker='.')
beta_275_nd = ax.scatter(beta_275_new[0,0],beta_275_new[0,1],color='g',marker='.')

ax.scatter(orig_dat[:,0],orig_dat[:,1],color='k',marker='o',s=0.1,alpha=0.1)
ax.scatter(beta_275_old[:,0],beta_275_old[:,1],color='b',marker='.',s=0.1,alpha=0.1)
ax.scatter(orig_new_dat[:,0],orig_new_dat[:,1],color='r',marker='.',s=0.1,alpha=0.1)
ax.scatter(beta_275_new[:,0],beta_275_new[:,1],color='g',marker='.',s=0.1,alpha=0.1)



save_filename=os.path.join(plots_dir,'umap_orig_beta275_comp.pdf')
#print("total number of key points plotted: "+ str(np.shape(loss_comp_latents)[0]))
plt.legend([orig_embed_old,orig_embed_new,beta_275_od,beta_275_nd],['orig embed old data','orig embed new data','b=2.75 old data','b=2.75 new data'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

##############
#second: 2.0

ax = plt.gca()
ol_inds = np.hstack([old_data_inds[0],old_data_inds[-3]])
ol_data = np.vstack([old_data_latents[0],old_data_latents[-3]])
ol_inds[ol_inds > 1] = 2
nl_inds = np.hstack([new_data_inds[0],new_data_inds[-3]])
nl_inds[nl_inds > 1] = 4
nl_inds[nl_inds == 1] = 3
nl_data = np.vstack([new_data_latents[0],new_data_latents[-3]])
all_inds = np.hstack([ol_inds, nl_inds])
print(np.unique(all_inds))

embed = transform.fit_transform(np.vstack([ol_data,nl_data]))
orig_dat = embed[all_inds == 1,:]
beta_20_old = embed[all_inds == 2,:]
orig_new_dat = embed[all_inds == 3,:]
beta_20_new = embed[all_inds == 4.,:]

orig_embed_old = ax.scatter(orig_dat[0,0],orig_dat[0,1],color='k',marker='o',)
beta_20_od = ax.scatter(beta_20_old[0,0],beta_20_old[0,1],color='b',marker='.')
orig_embed_new = ax.scatter(orig_new_dat[0,0],orig_new_dat[0,1],color='r',marker='.')
beta_20_nd = ax.scatter(beta_20_new[0,0],beta_20_new[0,1],color='g',marker='.')

ax.scatter(orig_dat[:,0],orig_dat[:,1],color='k',marker='o',s=0.1,alpha=0.1)
ax.scatter(beta_20_old[:,0],beta_20_old[:,1],color='b',marker='.',s=0.1,alpha=0.1)
ax.scatter(orig_new_dat[:,0],orig_new_dat[:,1],color='r',marker='.',s=0.1,alpha=0.1)
ax.scatter(beta_20_new[:,0],beta_20_new[:,1],color='g',marker='.',s=0.1,alpha=0.1)



save_filename=os.path.join(plots_dir,'umap_orig_beta20_comp.pdf')
#print("total number of key points plotted: "+ str(np.shape(loss_comp_latents)[0]))
plt.legend([orig_embed_old,orig_embed_new,beta_20_od,beta_20_nd],['orig embed old data','orig embed new data','b=2.0 old data','b=2.0 new data'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')
##############
# beta = 1.25

ax = plt.gca()
ol_inds = np.hstack([old_data_inds[0],old_data_inds[-4]])
ol_data = np.vstack([old_data_latents[0],old_data_latents[-4]])
ol_inds[ol_inds > 1] = 2
nl_inds = np.hstack([new_data_inds[0],new_data_inds[-4]])
nl_inds[nl_inds > 1] = 4
nl_inds[nl_inds == 1] = 3
nl_data = np.vstack([new_data_latents[0],new_data_latents[-4]])
all_inds = np.hstack([ol_inds, nl_inds])
print(np.unique(all_inds))

embed = transform.fit_transform(np.vstack([ol_data,nl_data]))
orig_dat = embed[all_inds == 1,:]
beta_125_old = embed[all_inds == 2,:]
orig_new_dat = embed[all_inds == 3,:]
beta_125_new = embed[all_inds == 4.,:]

orig_embed_old = ax.scatter(orig_dat[0,0],orig_dat[0,1],color='k',marker='o',)
beta_125_od = ax.scatter(beta_125_old[0,0],beta_125_old[0,1],color='b',marker='.')
orig_embed_new = ax.scatter(orig_new_dat[0,0],orig_new_dat[0,1],color='r',marker='.')
beta_125_nd = ax.scatter(beta_125_new[0,0],beta_125_new[0,1],color='g',marker='.')

ax.scatter(orig_dat[:,0],orig_dat[:,1],color='k',marker='o',s=0.1,alpha=0.1)
ax.scatter(beta_125_old[:,0],beta_125_old[:,1],color='b',marker='.',s=0.1,alpha=0.1)
ax.scatter(orig_new_dat[:,0],orig_new_dat[:,1],color='r',marker='.',s=0.1,alpha=0.1)
ax.scatter(beta_125_new[:,0],beta_125_new[:,1],color='g',marker='.',s=0.1,alpha=0.1)



save_filename=os.path.join(plots_dir,'umap_orig_beta125_comp.pdf')
#print("total number of key points plotted: "+ str(np.shape(loss_comp_latents)[0]))
plt.legend([orig_embed_old,orig_embed_new,beta_125_od,beta_125_nd],['orig embed old data','orig embed new data','b=1.25 old data','b=1.25 new data'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')
##############
# beta = 0.5
ax = plt.gca()
ol_inds = np.hstack([old_data_inds[0],old_data_inds[-5]])
ol_data = np.vstack([old_data_latents[0],old_data_latents[-5]])
ol_inds[ol_inds > 1] = 2
nl_inds = np.hstack([new_data_inds[0],new_data_inds[-5]])
nl_inds[nl_inds > 1] = 4
nl_inds[nl_inds == 1] = 3
nl_data = np.vstack([new_data_latents[0],new_data_latents[-5]])
all_inds = np.hstack([ol_inds, nl_inds])
print(np.unique(all_inds))

embed = transform.fit_transform(np.vstack([ol_data,nl_data]))
orig_dat = embed[all_inds == 1,:]
beta_05_old = embed[all_inds == 2,:]
orig_new_dat = embed[all_inds == 3,:]
beta_05_new = embed[all_inds == 4.,:]

orig_embed_old = ax.scatter(orig_dat[0,0],orig_dat[0,1],color='k',marker='o',)
beta_05_od = ax.scatter(beta_05_old[0,0],beta_05_old[0,1],color='b',marker='.')
orig_embed_new = ax.scatter(orig_new_dat[0,0],orig_new_dat[0,1],color='r',marker='.')
beta_05_nd = ax.scatter(beta_05_new[0,0],beta_05_new[0,1],color='g',marker='.')

ax.scatter(orig_dat[:,0],orig_dat[:,1],color='k',marker='o',s=0.1,alpha=0.1)
ax.scatter(beta_05_old[:,0],beta_05_old[:,1],color='b',marker='.',s=0.1,alpha=0.1)
ax.scatter(orig_new_dat[:,0],orig_new_dat[:,1],color='r',marker='.',s=0.1,alpha=0.1)
ax.scatter(beta_05_new[:,0],beta_05_new[:,1],color='g',marker='.',s=0.1,alpha=0.1)



save_filename=os.path.join(plots_dir,'umap_orig_beta05_comp.pdf')
#print("total number of key points plotted: "+ str(np.shape(loss_comp_latents)[0]))
plt.legend([orig_embed_old,orig_embed_new,beta_05_od,beta_05_nd],['orig embed old data','orig embed new data','b=0.5 old data','b=0.5 new data'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

###############
#Plotting all old data latents

##############
ax = plt.gca()

print('Old Data Plot')
ol_inds = np.hstack(old_data_inds)
ol_data = np.vstack(old_data_latents)
#print(np.shape(ol_data))
#ol_data = transform.fit_transform(ol_data)
transform2 = PCA(n_components=2, copy=False, random_state=42)
ol_data = transform2.fit_transform(ol_data)
#print(np.shape(ol_data))
#print(np.shape(ol_inds))
orig_dat = ol_data[ol_inds == 1,:]
b_05_dat = ol_data[ol_inds == 2,:]
b_125_dat = ol_data[ol_inds == 3,:]
b_20_dat = ol_data[ol_inds == 4,:]
b_275_dat = ol_data[ol_inds == 5,:]
b_35_dat = ol_data[ol_inds == 6,:]

orig_embed = ax.scatter(orig_dat[0,0],orig_dat[0,1],color='k',marker='o')
#b_05 = ax.scatter(b_05_dat[0,0],b_05_dat[0,1],color='b',marker='.')
#b_125 = ax.scatter(b_125_dat[0,0],b_125_dat[0,1],color='r',marker='.')
#b_20 = ax.scatter(b_20_dat[0,0],b_20_dat[0,1],color='g',marker='.')
#b_275 = ax.scatter(b_275_dat[0,0],b_275_dat[0,1],color='c',marker='.')
b_35 = ax.scatter(b_35_dat[0,0],b_35_dat[0,1],color='m',marker='.')

ax.scatter(orig_dat[:,0],orig_dat[:,1],color='k',marker='o',s=0.1,alpha=0.1)
#ax.scatter(b_05_dat[:,0],b_05_dat[:,1],color='b',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_125_dat[:,0],b_125_dat[:,1],color='r',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_20_dat[:,0],b_20_dat[:,1],color='g',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_275_dat[:,0],b_275_dat[:,1],color='c',marker='.',s=0.1,alpha=0.1)
ax.scatter(b_35_dat[:,0],b_35_dat[:,1],color='m',marker='.',s=0.1,alpha=0.1)

save_filename=os.path.join(plots_dir,'pca_old_latents_betas_35.pdf')
#print("total number of key points plotted: "+ str(np.shape(loss_comp_latents)[0]))
plt.legend([orig_embed,b_35],['orig embed','b=3.5'])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

###########################################

ax = plt.gca()

print('Old Data Plot')

orig_embed = ax.scatter(orig_dat[0,0],orig_dat[0,1],color='k',marker='o')
#b_05 = ax.scatter(b_05_dat[0,0],b_05_dat[0,1],color='b',marker='.')
#b_125 = ax.scatter(b_125_dat[0,0],b_125_dat[0,1],color='r',marker='.')
#b_20 = ax.scatter(b_20_dat[0,0],b_20_dat[0,1],color='g',marker='.')
b_275 = ax.scatter(b_275_dat[0,0],b_275_dat[0,1],color='c',marker='.')
#b_35 = ax.scatter(b_35_dat[0,0],b_35_dat[0,1],color='m',marker='.')

ax.scatter(orig_dat[:,0],orig_dat[:,1],color='k',marker='o',s=0.1,alpha=0.1)
#ax.scatter(b_05_dat[:,0],b_05_dat[:,1],color='b',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_125_dat[:,0],b_125_dat[:,1],color='r',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_20_dat[:,0],b_20_dat[:,1],color='g',marker='.',s=0.1,alpha=0.1)
ax.scatter(b_275_dat[:,0],b_275_dat[:,1],color='c',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_35_dat[:,0],b_35_dat[:,1],color='m',marker='.',s=0.1,alpha=0.1)

save_filename=os.path.join(plots_dir,'pca_old_latents_betas_275.pdf')
#print("total number of key points plotted: "+ str(np.shape(loss_comp_latents)[0]))
plt.legend([orig_embed,b_35],['orig embed','b=2.75'])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

##########################################

ax = plt.gca()

print('Old Data Plot')

orig_embed = ax.scatter(orig_dat[0,0],orig_dat[0,1],color='k',marker='o')
#b_05 = ax.scatter(b_05_dat[0,0],b_05_dat[0,1],color='b',marker='.')
#b_125 = ax.scatter(b_125_dat[0,0],b_125_dat[0,1],color='r',marker='.')
b_20 = ax.scatter(b_20_dat[0,0],b_20_dat[0,1],color='g',marker='.')
#b_275 = ax.scatter(b_275_dat[0,0],b_275_dat[0,1],color='c',marker='.')
#b_35 = ax.scatter(b_35_dat[0,0],b_35_dat[0,1],color='m',marker='.')

ax.scatter(orig_dat[:,0],orig_dat[:,1],color='k',marker='o',s=0.1,alpha=0.1)
#ax.scatter(b_05_dat[:,0],b_05_dat[:,1],color='b',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_125_dat[:,0],b_125_dat[:,1],color='r',marker='.',s=0.1,alpha=0.1)
ax.scatter(b_20_dat[:,0],b_20_dat[:,1],color='g',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_275_dat[:,0],b_275_dat[:,1],color='c',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_35_dat[:,0],b_35_dat[:,1],color='m',marker='.',s=0.1,alpha=0.1)

save_filename=os.path.join(plots_dir,'pca_old_latents_betas_20.pdf')
#print("total number of key points plotted: "+ str(np.shape(loss_comp_latents)[0]))
plt.legend([orig_embed,b_35],['orig embed','b=2.0'])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

############################

ax = plt.gca()

print('Old Data Plot')

orig_embed = ax.scatter(orig_dat[0,0],orig_dat[0,1],color='k',marker='o')
#b_05 = ax.scatter(b_05_dat[0,0],b_05_dat[0,1],color='b',marker='.')
b_125 = ax.scatter(b_125_dat[0,0],b_125_dat[0,1],color='r',marker='.')
#b_20 = ax.scatter(b_20_dat[0,0],b_20_dat[0,1],color='g',marker='.')
#b_275 = ax.scatter(b_275_dat[0,0],b_275_dat[0,1],color='c',marker='.')
#b_35 = ax.scatter(b_35_dat[0,0],b_35_dat[0,1],color='m',marker='.')

ax.scatter(orig_dat[:,0],orig_dat[:,1],color='k',marker='o',s=0.1,alpha=0.1)
#ax.scatter(b_05_dat[:,0],b_05_dat[:,1],color='b',marker='.',s=0.1,alpha=0.1)
ax.scatter(b_125_dat[:,0],b_125_dat[:,1],color='r',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_20_dat[:,0],b_20_dat[:,1],color='g',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_275_dat[:,0],b_275_dat[:,1],color='c',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_35_dat[:,0],b_35_dat[:,1],color='m',marker='.',s=0.1,alpha=0.1)

save_filename=os.path.join(plots_dir,'pca_old_latents_betas_125.pdf')
#print("total number of key points plotted: "+ str(np.shape(loss_comp_latents)[0]))
plt.legend([orig_embed,b_35],['orig embed','b=1.25'])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')


###############################################################

ax = plt.gca()

print('Old Data Plot')

orig_embed = ax.scatter(orig_dat[0,0],orig_dat[0,1],color='k',marker='o')
b_05 = ax.scatter(b_05_dat[0,0],b_05_dat[0,1],color='b',marker='.')
#b_125 = ax.scatter(b_125_dat[0,0],b_125_dat[0,1],color='r',marker='.')
#b_20 = ax.scatter(b_20_dat[0,0],b_20_dat[0,1],color='g',marker='.')
#b_275 = ax.scatter(b_275_dat[0,0],b_275_dat[0,1],color='c',marker='.')
#b_35 = ax.scatter(b_35_dat[0,0],b_35_dat[0,1],color='m',marker='.')

ax.scatter(orig_dat[:,0],orig_dat[:,1],color='k',marker='o',s=0.1,alpha=0.1)
ax.scatter(b_05_dat[:,0],b_05_dat[:,1],color='b',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_125_dat[:,0],b_125_dat[:,1],color='r',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_20_dat[:,0],b_20_dat[:,1],color='g',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_275_dat[:,0],b_275_dat[:,1],color='c',marker='.',s=0.1,alpha=0.1)
#ax.scatter(b_35_dat[:,0],b_35_dat[:,1],color='m',marker='.',s=0.1,alpha=0.1)

save_filename=os.path.join(plots_dir,'pca_old_latents_betas_05.pdf')
#print("total number of key points plotted: "+ str(np.shape(loss_comp_latents)[0]))
plt.legend([orig_embed,b_35],['orig embed','b=0.5'])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

####################
# Plotting all new data latents
####################

ax = plt.gca()

print('New Data Plot')
nl_inds = np.hstack(new_data_inds)
nl_data = np.vstack(new_data_latents)

nl_data = transform.fit_transform(nl_data)

orig_dat = nl_data[nl_inds == 1,:]
b_05_dat = nl_data[nl_inds == 2,:]
b_125_dat = nl_data[nl_inds == 3,:]
b_20_dat = nl_data[nl_inds == 4,:]
b_275_dat = nl_data[nl_inds == 5,:]
b_35_dat = nl_data[nl_inds == 6,:]

orig_embed = ax.scatter(orig_dat[0,0],orig_dat[0,1],color='k',marker='o')
b_05 = ax.scatter(b_05_dat[0,0],b_05_dat[0,1],color='b',marker='.')
b_125 = ax.scatter(b_125_dat[0,0],b_125_dat[0,1],color='r',marker='.')
b_20 = ax.scatter(b_20_dat[0,0],b_20_dat[0,1],color='g',marker='.')
b_275 = ax.scatter(b_275_dat[0,0],b_275_dat[0,1],color='c',marker='.')
b_35 = ax.scatter(b_35_dat[0,0],b_35_dat[0,1],color='m',marker='.')

ax.scatter(orig_dat[:,0],orig_dat[:,1],color='k',marker='o',s=0.1,alpha=0.1)
ax.scatter(b_05_dat[:,0],b_05_dat[:,1],color='b',marker='.',s=0.1,alpha=0.1)
ax.scatter(b_125_dat[:,0],b_125_dat[:,1],color='r',marker='.',s=0.1,alpha=0.1)
ax.scatter(b_20_dat[:,0],b_20_dat[:,1],color='g',marker='.',s=0.1,alpha=0.1)
ax.scatter(b_275_dat[:,0],b_275_dat[:,1],color='c',marker='.',s=0.1,alpha=0.1)
ax.scatter(b_35_dat[:,0],b_35_dat[:,1],color='m',marker='.',s=0.1,alpha=0.1)

save_filename=os.path.join(plots_dir,'umap_new_latents_betas.pdf')
#print("total number of key points plotted: "+ str(np.shape(loss_comp_latents)[0]))
plt.legend([orig_embed,b_05,b_125,b_20,b_275,b_35],['orig embed','b=0.5','b=1.25','b=2.0','b=2.75','b=3.5'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

######
# MSE Plot
#########
latents = np.vstack(loss_comp_latents)
mse_b_05_orig = np.sum((latents[embed_inds == 1,:] - latents[embed_inds == 2,:])**2,axis=1)/z_dim
mse_b_125_orig = np.sum((latents[embed_inds == 1,:] - latents[embed_inds == 3,:])**2,axis=1)/z_dim
mse_b_2_orig = np.sum((latents[embed_inds == 1,:] - latents[embed_inds == 4,:])**2,axis=1)/z_dim
mse_b_275_orig = np.sum((latents[embed_inds == 1,:] - latents[embed_inds == 5,:])**2,axis=1)/z_dim
mse_b_35_orig = np.sum((latents[embed_inds == 1,:] - latents[embed_inds == 6,:])**2,axis=1)/z_dim
mse_dict = {'mse_beta05':mse_b_05_orig, 'mse_beta125':mse_b_125_orig, 'mse_beta_2':mse_b_2_orig,\
            'mse_beta275':mse_b_275_orig, 'mse_beta35':mse_b_35_orig}
mse_dat = pd.DataFrame(data=mse_dict,dtype=np.float64)
#mse_dat["id"] = mse_dat.index
#mse_dat = pd.wide_to_long(mse_dat,'mse',i='id',j='mse')
sb.set_theme(style="whitegrid")

#ax = sb.boxplot(x=[''])
save_filename=os.path.join(plots_dir,'mseplot_betas.pdf')
ax = sb.swarmplot(data = mse_dat,size=2)
fig = ax.get_figure()
fig.savefig(save_filename)
plt.close('all')
'''
'''

all_new_points = np.vstack((np.vstack(loss_comp_latents),both_loss_old_latents))

latents_inds = 5*np.ones((len(both_loss_old_latents),))

all_inds_both = np.hstack((ind_vec,latents_inds))
print(np.shape(ind_vec))
print(np.shape(latents_inds))
print("Running UMAP... (n="+str(len(all_new_points))+")")

embed = transform.fit_transform(all_new_points)
print('Done!')


# Plot 1 ############################################
print('UMAP Plot')
ax = plt.gca()

old_embed = embed[all_inds_both == 2,:]
both_embed = embed[all_inds_both == 1,:]
orig_embed = embed[all_inds_both == 4,:]
ax.scatter(embed[all_inds_both == 5,0],embed[all_inds_both == 5,1],color = 'blue',s=0.1,alpha=0.1)
#for ind1 in range(n_encoded):
#    ind2 = -n_encoded + ind1
#    ax.plot([embed[ind1,0],embed[ind2,0]],[embed[ind1,1],embed[ind2,1]],'--',linewidth=0.5)
for ind in range(len(both_embed)):
    ax.plot([old_embed[ind,0],orig_embed[ind,0]],\
            [old_embed[ind,1],orig_embed[ind,1]], '--k',linewidth=0.5)
#1 both loss, 2 old loss, 3 new loss, 4 orig latents
#a1 = ax.scatter(embed[all_inds_both == 1,0],embed[all_inds_both == 1,1],color='k',marker='o',s=4.5,alpha=1)
a2 = ax.scatter(embed[all_inds_both == 2,0],embed[all_inds_both == 2,1],color='k',marker='v',s=4.5,alpha=1)
#a3 = ax.scatter(embed[all_inds_both == 3,0],embed[all_inds_both == 3,1],color='k',marker='x',s=4.5,alpha=1)
a4 = ax.scatter(embed[all_inds_both == 4,0],embed[all_inds_both == 4,1],color='g',marker='.',s=4.5,alpha=1)

#ax.scatter(embed[-25:,0], embed[-25:,1], 'rx',s=1.5,alpha=1)
save_filename=os.path.join(plots_dir,'umap_bothLoss_old_orig_con.pdf')
#print("total number of key points plotted: "+ str(np.shape(all_orig_latents)[0]))
plt.legend([a1,a2,a4],['both loss umap_key_points_orig_both_con','old loss keys','original keys'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

# Plot 2  ################################################
ax = plt.gca()

ax.scatter(embed[all_inds_both == 5,0],embed[all_inds_both == 5,1],color = 'blue',s=0.1,alpha=0.1)
#for ind1 in range(n_encoded):
#    ind2 = -n_encoded + ind1
#    ax.plot([embed[ind1,0],embed[ind2,0]],[embed[ind1,1],embed[ind2,1]],'--',linewidth=0.5)
for ind in range(len(both_embed)):
    ax.plot([both_embed[ind,0],orig_embed[ind,0]],\
            [both_embed[ind,1],orig_embed[ind,1]], '--k',linewidth=0.5)
#1 both loss, 2 old loss, 3 new loss, 4 orig latents
a1 = ax.scatter(embed[all_inds_both == 1,0],embed[all_inds_both == 1,1],color='k',marker='o',s=4.5,alpha=1)
#a2 = ax.scatter(embed[all_inds_both == 2,0],embed[all_inds_both == 2,1],color='k',marker='v',s=4.5,alpha=1)
#a3 = ax.scatter(embed[all_inds_both == 3,0],embed[all_inds_both == 3,1],color='k',marker='x',s=4.5,alpha=1)
a4 = ax.scatter(embed[all_inds_both == 4,0],embed[all_inds_both == 4,1],color='g',marker='.',s=4.5,alpha=1)

#ax.scatter(embed[-25:,0], embed[-25:,1], 'rx',s=1.5,alpha=1)
save_filename=os.path.join(plots_dir,'umap_bothLoss_both_orig_con.pdf')
#print("total number of key points plotted: "+ str(np.shape(all_orig_latents)[0]))
plt.legend([a1,a2,a4],['both loss keys','old loss keys','original keys'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')
############
# plotting all key points with old data latents, old loss space


# Plot 1: both loss, orig keys connected
###########


all_old_points = np.vstack((np.vstack(loss_comp_latents),old_loss_old_latents))
print("Running UMAP... (n="+str(len(all_old_points))+")")


embed = transform.fit_transform(all_old_points)
latents_inds = 5*np.ones((len(old_loss_old_latents),))

all_inds_old = np.hstack((ind_vec,latents_inds))
print('Done!')
ax = plt.gca()

print('UMAP Plot')

both_embed = embed[all_inds_old == 1]
orig_embed = embed[all_inds_old == 4]
#### Plot 1 ####
ax = plt.gca()
ax.scatter(embed[all_inds_old == 5,0],embed[all_inds_old == 5,1],color = 'blue',s=0.1,alpha=0.1)
#for ind1 in range(n_encoded):
#ind2 = -n_encoded + ind1
#ax.plot([embed[ind1,0],embed[ind2,0]],[embed[ind1,1],embed[ind2,1]],'--',linewidth=0.5)

for ind in range(len(both_embed)):
    ax.plot([both_embed[ind,0],orig_embed[ind,0]],[both_embed[ind,1],orig_embed[ind,1]],'--k',linewidth=0.5)
#1 both loss, 2 old loss, 3 new loss, 4 orig latents
a1 = ax.scatter(embed[all_inds_old == 1,0],embed[all_inds_old == 1,1],color='k',marker='o',s=4.5,alpha=1)
a2 = ax.scatter(embed[all_inds_old == 2,0],embed[all_inds_old == 2,1],color='k',marker='v',s=4.5,alpha=1)
#a3 = ax.scatter(embed[all_inds_old == 3,0],embed[all_inds_old == 3,1],color='k',marker='x',s=4.5,alpha=1)
a4 = ax.scatter(embed[all_inds_old == 4,0],embed[all_inds_old == 4,1],color='g',marker='.',s=4.5,alpha=1)

#im6 = ax.scatter(embed[-25:,0], embed[-25:,1], color= 'red',s=1.5,alpha=1)
save_filename=os.path.join(plots_dir,'umap_key_points_old_loss.pdf')
#print("total number of key points plotted: "+ str(np.shape(all_old_points)[0]))
plt.legend([a1,a2,a4],['both loss','old loss','old model points'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')


#########
# Plotting all key points with old data latents, new loss space
##########
all_new_points = np.vstack((np.vstack(loss_comp_latents),new_loss_old_latents))
print("Running UMAP... (n="+str(len(all_new_points))+")")
latents_inds = 5*np.ones((len(new_loss_old_latents),))

all_inds_new = np.hstack((ind_vec,latents_inds))
embed = transform.fit_transform(all_new_points)
print('Done!')
ax = plt.gca()

print('UMAP Plot')
ax = plt.gca()
ax.scatter(embed[all_inds_new == 5,0],embed[all_inds_new == 5,1],color = 'blue',s=0.1,alpha=0.1)
#for ind1 in range(n_encoded):
#    ind2 = -n_encoded + ind1
#    ax.plot([embed[ind1,0],embed[ind2,0]],[embed[ind1,1],embed[ind2,1]],'--',linewidth=0.5)

#1 both loss, 2 old loss, 3 new loss, 4 orig latents
a1 = ax.scatter(embed[all_inds_new == 1,0],embed[all_inds_new == 1,1],color='k',marker='o',s=4.5,alpha=1)
a2 = ax.scatter(embed[all_inds_new == 2,0],embed[all_inds_new == 2,1],color='k',marker='v',s=4.5,alpha=1)
#a3 = ax.scatter(embed[all_inds_new == 3,0],embed[all_inds_new == 3,1],color='k',marker='x',s=4.5,alpha=1)
a4 = ax.scatter(embed[all_inds_new == 4,0],embed[all_inds_old == 4,1],color='g',marker='.',s=4.5,alpha=1)

#im6 = ax.scatter(embed[-25:,0], embed[-25:,1], color= 'red',s=1.5,alpha=1)
save_filename=os.path.join(plots_dir,'umap_old_latents_new_loss.pdf')
#print("total number of key points plotted: "+ str(np.shape(all_old_points)[0]))
plt.legend([a1,a2,a4],['both loss','old loss','old model points'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')


################
# Finally, all key points in the original model space
################
all_old_model_space = np.vstack((np.vstack(loss_comp_latents),old_model_old_latents))
print("Running UMAP... (n="+str(len(all_old_model_space))+")")
latents_inds = 5*np.ones((len(old_model_old_latents),))

all_inds_old_space = np.hstack((ind_vec,latents_inds))
embed = transform.fit_transform(all_old_model_space)
print('Done!')
ax = plt.gca()

print('UMAP Plot')
ax = plt.gca()
ax.scatter(embed[all_inds_old_space == 5,0],embed[all_inds_old_space == 5,1],color = 'blue',s=0.1,alpha=0.1)
#for ind1 in range(n_encoded):
#    ind2 = -n_encoded + ind1
#    ax.plot([embed[ind1,0],embed[ind2,0]],[embed[ind1,1],embed[ind2,1]],'--',linewidth=0.5)

#1 both loss, 2 old loss, 3 new loss, 4 orig latents
a1 = ax.scatter(embed[all_inds_old_space == 1,0],embed[all_inds_old_space == 1,1],color='k',marker='o',s=4.5,alpha=1)
a2 = ax.scatter(embed[all_inds_old_space == 2,0],embed[all_inds_old_space == 2,1],color='k',marker='v',s=4.5,alpha=1)
#a3 = ax.scatter(embed[all_inds_new == 3,0],embed[all_inds_new == 3,1],color='k',marker='x',s=4.5,alpha=1)
a4 = ax.scatter(embed[all_inds_old_space == 4,0],embed[all_inds_old_space == 4,1],color='g',marker='.',s=4.5,alpha=1)

#im6 = ax.scatter(embed[-25:,0], embed[-25:,1], color= 'red',s=1.5,alpha=1)
save_filename=os.path.join(plots_dir,'umap_old_model_space_keypoints.pdf')
#print("total number of key points plotted: "+ str(np.shape(all_old_points)[0]))
plt.legend([a1,a2,a4],['both loss','old loss','old model points'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')


#########
# bird 1, bird 2 latents through old model - how do these look compared to all the key points?
#########
all_data_old_space = np.vstack((np.vstack(loss_comp_latents),old_model_old_latents,old_model_new_latents))
old_lat_inds = 5*np.ones((len(old_model_old_latents),))
new_lat_inds = 6*np.ones((len(old_model_new_latents),))

all_inds = np.hstack((ind_vec,old_lat_inds,new_lat_inds))
print("Running UMAP... (n="+str(len(all_data_old_space))+")")
embed = transform.fit_transform(all_data_old_space)

ax = plt.gca()
old = ax.scatter(embed[all_inds == 5,0],embed[all_inds == 5,1],color='blue',s=0.1,alpha=0.1)
new = ax.scatter(embed[all_inds == 6,0],embed[all_inds == 6,1],color='red', s=0.1,alpha=0.1)

#1 both loss, 2 old loss, 3 new loss, 4 orig latents
a1 =  ax.scatter(embed[all_inds == 1,0],embed[all_inds == 1,1], color='k',marker='o',s=4.5,alpha=1)
a2 = ax.scatter(embed[all_inds == 2,0],embed[all_inds == 2,1],color='k',marker='v',s=4.5,alpha=1)
#a3 = ax.scatter(embed[all_inds == 3,0],embed[all_inds == 3,1],color='k',marker='x',s=4.5,alpha=1)
a4 = ax.scatter(embed[all_inds == 4,0],embed[all_inds == 4,1],color='g',marker='.',s=4.5,alpha=1)

save_filename=os.path.join(plots_dir,'umap_old_space_all_points.pdf')
#print("total number of key points plotted: "+ str(np.shape(all_old_points)[0]))
plt.legend([old,new,a2,a1,a4],['bird1 old loss','bird2 old loss','old loss keys','both loss keys','original keys'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('Bird 1, Bird 2, Key Points through Original Model')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

###############
# bird 1, bird 2 latents through old loss model - how distorted does bird 1 get if you only give it these 100 points?
###############
all_data_old_space = np.vstack((np.vstack(loss_comp_latents),old_loss_old_latents,old_loss_new_latents))
old_lat_inds = 5*np.ones((len(old_loss_old_latents),))
new_lat_inds = 6*np.ones((len(old_loss_new_latents),))

all_inds = np.hstack((ind_vec,old_lat_inds,new_lat_inds))
print("Running UMAP... (n="+str(len(all_data_old_space))+")")
embed = transform.fit_transform(all_data_old_space)

ax = plt.gca()
old = ax.scatter(embed[all_inds == 5,0],embed[all_inds == 5,1],color='blue',s=0.1,alpha=0.1)
new = ax.scatter(embed[all_inds == 6,0],embed[all_inds == 6,1],color='red', s=0.1,alpha=0.1)

#1 both loss, 2 old loss, 3 new loss, 4 orig latents
a1 =  ax.scatter(embed[all_inds == 1,0],embed[all_inds == 1,1], color='k',marker='o',s=4.5,alpha=1)
a2 = ax.scatter(embed[all_inds == 2,0],embed[all_inds == 2,1],color='k',marker='v',s=4.5,alpha=1)
#a3 = ax.scatter(embed[all_inds == 3,0],embed[all_inds == 3,1],color='k',marker='x',s=4.5,alpha=1)
a4 = ax.scatter(embed[all_inds == 4,0],embed[all_inds == 4,1],color='g',marker='.',s=4.5,alpha=1)

save_filename=os.path.join(plots_dir,'umap_old_loss_space_all_points.pdf')
#print("total number of key points plotted: "+ str(np.shape(all_old_points)[0]))
plt.legend([old,new,a2,a1,a4],['bird1 old loss','bird2 old loss','old loss keys','both loss keys','original keys'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('Bird 1, Bird 2, key points through old loss model')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')
###############
# bird 1, bird 2 latents through both loss model - how do things change?
###############
all_data_old_space = np.vstack((np.vstack(loss_comp_latents),both_loss_old_latents,both_loss_new_latents))
old_lat_inds = 5*np.ones((len(both_loss_old_latents),))
new_lat_inds = 6*np.ones((len(both_loss_new_latents),))

all_inds = np.hstack((ind_vec,old_lat_inds,new_lat_inds))
print("Running UMAP... (n="+str(len(all_data_old_space))+")")
embed = transform.fit_transform(all_data_old_space)

ax = plt.gca()
old = ax.scatter(embed[all_inds == 5,0],embed[all_inds == 5,1],color='blue',s=0.1,alpha=0.1)
new = ax.scatter(embed[all_inds == 6,0],embed[all_inds == 6,1],color='red', s=0.1,alpha=0.1)

#1 both loss, 2 old loss, 3 new loss, 4 orig latents
a1 =  ax.scatter(embed[all_inds == 1,0],embed[all_inds == 1,1], color='k',marker='o',s=4.5,alpha=1)
a2 = ax.scatter(embed[all_inds == 2,0],embed[all_inds == 2,1],color='k',marker='v',s=4.5,alpha=1)
#a3 = ax.scatter(embed[all_inds == 3,0],embed[all_inds == 3,1],color='k',marker='x',s=4.5,alpha=1)
a4 = ax.scatter(embed[all_inds == 4,0],embed[all_inds == 4,1],color='g',marker='.',s=4.5,alpha=1)

save_filename=os.path.join(plots_dir,'umap_both_loss_space_all_points.pdf')
#print("total number of key points plotted: "+ str(np.shape(all_old_points)[0]))
plt.legend([old,new,a2,a1,a4],['bird 1','bird 2','old loss keys','both loss keys','original keys'])
plt.title('Bird 1, bird 2, key points through both loss modell')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')
###############
# bird 2 latents through old loss, both loss, new loss model - how do these all compare?
###############
all_data_old_space = np.vstack((np.vstack(loss_comp_latents),old_loss_new_latents,\
                              new_loss_new_latents,both_loss_new_latents))
old_loss_inds = 5*np.ones((len(old_loss_new_latents),))
new_loss_inds = 6*np.ones((len(new_loss_new_latents),))
both_loss_inds = 7*np.ones((len(both_loss_new_latents),))

all_inds = np.hstack((ind_vec,old_loss_inds,new_loss_inds,both_loss_inds))
print("Running UMAP... (n="+str(len(all_data_old_space))+")")
embed = transform.fit_transform(all_data_old_space)

ax = plt.gca()
old_loss = ax.scatter(embed[all_inds == 5,0],embed[all_inds == 5,1],color='blue',s=0.1,alpha=0.1)
new_loss = ax.scatter(embed[all_inds == 6,0],embed[all_inds == 6,1],color='red', s=0.1,alpha=0.1)
both_loss = ax.scatter(embed[all_inds == 7,0],embed[all_inds == 7,1],color='orange',s=0.1,alpha=0.1)
#1 both loss, 2 old loss, 3 new loss, 4 orig latents
#a1 =  ax.scatter(embed[all_inds == 1,0],embed[all_inds == 1,1], colors='k',marker='o',s=4.5,alpha=1)
#a2 = ax.scatter(embed[all_inds == 2,0],embed[all_inds == 2,1],color='k',marker='v',s=4.5,alpha=1)
#a3 = ax.scatter(embed[all_inds == 3,0],embed[all_inds == 3,1],color='k',marker='x',s=4.5,alpha=1)
#a4 = ax.scatter(embed[all_inds == 4,0],embed[all_inds == 4,1],color='g',marker='.',s=4.5,alpha=1)

save_filename=os.path.join(plots_dir,'umap_all_new_points.pdf')
#print("total number of key points plotted: "+ str(np.shape(all_old_points)[0]))
plt.legend([old_loss,new_loss,both_loss],['old loss bird 2','new loss bird 2','both loss bird 2'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('Bird 2 latents through all models')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

###########
# bird 1 latents in all loss cases
############
#### plot 1: old, orig relative to each other
all_data_old_space = np.vstack((np.vstack(loss_comp_latents),old_loss_old_latents,\
                              new_loss_old_latents,both_loss_old_latents, \
                              old_model_old_latents))
old_loss_inds = 5*np.ones((len(old_loss_old_latents),))
new_loss_inds = 6*np.ones((len(new_loss_old_latents),))
both_loss_inds = 7*np.ones((len(both_loss_old_latents),))
old_model_latents = 8*np.ones((len(old_model_old_latents),))

all_inds = np.hstack((ind_vec,old_loss_inds,new_loss_inds,both_loss_inds,old_model_latents))
print("Running UMAP... (n="+str(len(all_data_old_space))+")")
embed = transform.fit_transform(all_data_old_space)

ax = plt.gca()
old_loss = ax.scatter(embed[all_inds == 5,0],embed[all_inds == 5,1],color='blue',s=0.1,alpha=0.1)
new_loss = ax.scatter(embed[all_inds == 6,0],embed[all_inds == 6,1],color='red', s=0.1,alpha=0.1)
both_loss = ax.scatter(embed[all_inds == 7,0],embed[all_inds == 7,1],color='orange',s=0.1,alpha=0.1)
old_latents = ax.scatter(embed[all_inds == 8,0],embed[all_inds == 8,1],color='green',s=0.1,alpha=0.1)
#1 both loss, 2 old loss, 3 new loss, 4 orig latents
#a1 =  ax.scatter(embed[all_inds == 1,0],embed[all_inds == 1,1], colors='k',marker='o',s=4.5,alpha=1)
#a2 = ax.scatter(embed[all_inds == 2,0],embed[all_inds == 2,1],color='k',marker='v',s=4.5,alpha=1)
#a3 = ax.scatter(embed[all_inds == 3,0],embed[all_inds == 3,1],color='k',marker='x',s=4.5,alpha=1)
#a4 = ax.scatter(embed[all_inds == 4,0],embed[all_inds == 4,1],color='g',marker='.',s=4.5,alpha=1)

save_filename=os.path.join(plots_dir,'umap_all_old_points.pdf')
#print("total number of key points plotted: "+ str(np.shape(all_old_points)[0]))
plt.legend([old_loss,new_loss,both_loss,old_latents],\
            ['old loss bird 1','new loss bird 1','both loss bird 1','original latents bird 1'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('Bird 2 latents through all models')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

#################
# Bird 1, bird 2 latents in both loss space, old loss space, orig space
#################

all_data = np.vstack((old_loss_old_latents,\
                              old_loss_new_latents,both_loss_old_latents, \
                              both_loss_new_latents, old_model_old_latents, \
                              old_model_new_latents))
old_loss_old_inds = np.ones((len(old_loss_old_latents),))
old_loss_new_inds = 2*np.ones((len(old_loss_new_latents),))
both_loss_old_inds = 3*np.ones((len(both_loss_old_latents),))
both_loss_new_inds = 4*np.ones((len(both_loss_new_latents),))
orig_space_old_inds = 5*np.ones((len(old_model_old_latents),))
orig_space_new_inds = 6*np.ones((len(old_model_new_latents),))

all_inds = np.hstack((old_loss_old_inds,old_loss_new_inds,both_loss_old_inds,\
                      both_loss_new_inds,orig_space_old_inds,orig_space_new_inds))

print("Running UMAP... (n="+str(len(all_data))+")")

embed = transform.fit_transform(all_data)
print("Done!")
ax = plt.gca()
old_old = ax.scatter(embed[all_inds == 1,0],embed[all_inds == 1,1],s=0.1,alpha=0.3)
old_new = ax.scatter(embed[all_inds == 2,0],embed[all_inds == 2,1], s=0.1,alpha=0.3)
both_old = ax.scatter(embed[all_inds == 3,0],embed[all_inds == 3,1],s=0.1,alpha=0.3)
both_new = ax.scatter(embed[all_inds == 4,0],embed[all_inds == 4,1],s=0.1,alpha=0.3)

orig_old = ax.scatter(embed[all_inds == 5,0],embed[all_inds == 5,1],s=0.1,alpha=0.3)
orig_new = ax.scatter(embed[all_inds == 6,0],embed[all_inds == 6,1],s=0.1,alpha=0.3)

save_filename=os.path.join(plots_dir,'umap_all_points.pdf')
#print("total number of key points plotted: "+ str(np.shape(all_old_points)[0]))
plt.legend([old_old,old_new,both_old,both_new,orig_old,orig_new],\
            ['old old','old new','both old','both new','orig old','orig new'])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('Bird 2 latents through all models')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')


'''
