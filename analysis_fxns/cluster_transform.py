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

from ava.data.data_container import DataContainer
from ava.models.vae import VAE


from ava.models.vae_dataset import get_syllable_partition, get_syllable_data_loaders
from ava.plotting.latent_projection import latent_projection_plot_DC
from ava.plotting.grid_plot import grid_plot

######
# steps:
# 1. get bird 1 model, bird 2 model
# 2. get bird 1 centroids, bird 2 centroids
# 3. get bird 2 centroids through bird 1 model
# 4. look at distortion/map between bird 1, bird 2 models

root = '/home/mmartinez/autoencoded-vocal-analysis/ava_test3'

trainDays = ['100']
testDays = ['100']

animals_288 = ['blu288']
animals_258 = ['blu258']
plots_dir = os.path.join(root,'kmeans_plot')
if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)

### directories for analysis files - separating between test, train, and combined
#audio dirs
audio_dirs_258 = [os.path.join(root,animal,'audio',day) for animal in animals_258 for day in trainDays]
audio_dirs_288 = [os.path.join(root,animal,'audio',day) for animal in animals_288 for day in testDays]

#segmented syllable dirs
segment_dirs_258 = [os.path.join(root,animal,'segs',day) for animal in animals_258 for day in trainDays]
segment_dirs_288 = [os.path.join(root,animal,'segs',day) for animal in animals_288 for day in testDays]

#spectrogram dirs
spec_dirs_258 = [os.path.join(root,animal,'h5s',day) for animal in animals_258 for day in trainDays]
spec_dirs_288 = [os.path.join(root,animal,'h5s',day) for animal in animals_288 for day in testDays]

#projection dirs
proj_dirs_258 = [os.path.join(root,animal,'proj_means_258' + str(0),day) for animal in animals_258 for day in testDays]
proj_dirs_288 = [os.path.join(root,animal,'proj_means_288' + str(1),day) for animal in animals_288 for day in testDays]
proj_dirs_288_258 = [os.path.join(root,animal,'proj_means_288_258',day) for animal in animals_258 for day in testDays]
proj_dirs_258_288 = [os.path.join(root,animal,'proj_means_258_288',day) for animal in animals_288 for day in testDays]
#trained models
model_258_fname =  os.path.join(root,'model_258_0', 'checkpoint_100.tar')
model_288_fname = os.path.join(root,'model_288_0', 'checkpoint_100.tar')

#data containers
# first number: data run through model. second number: model number
model_258_dc = DataContainer(projection_dirs=proj_dirs_258,\
            audio_dirs=audio_dirs_258, spec_dirs=spec_dirs_258, model_filename = model_258_fname)
model_288_dc = DataContainer(projection_dirs=proj_dirs_288,\
            audio_dirs=audio_dirs_288, spec_dirs=spec_dirs_288, model_filename = model_288_fname)
model_258_288_dc = DataContainer(projection_dirs=proj_dirs_288_258,\
            audio_dirs=audio_dirs_258, spec_dirs=spec_dirs_258, model_filename = model_288_fname)
model_288_258_dc = DataContainer(projection_dirs=proj_dirs_258_288,\
            audio_dirs=audio_dirs_288, spec_dirs=spec_dirs_288, model_filename = model_258_fname)

lat_means_258 = model_258_dc.request('latent_means')
lat_means_288 = model_288_dc.request('latent_means')
lat_means_258_288 = model_258_288_dc.request('latent_means')
lat_means_288_258 = model_288_258_dc.request('latent_means')


#td_258_all = []
#td_288_all = []


#for n in range(1,15):
km_258_lat = KMeans(n_clusters=25).fit(lat_means_258)
km_288_lat = KMeans(n_clusters=25).fit(lat_means_288)
km_258_288_lat = KMeans(n_clusters=25).fit(lat_means_258_288)
km_288_258_lat = KMeans(n_clusters=25).fit(lat_means_288_258)


#    td_258 = copy.copy(km_258_lat.inertia_)
#    td_288 = copy.copy(km_288_lat.inertia_)

#    td_258_all.append(td_258)
#    td_288_all.append(td_288)

km_258_centroids = km_258_lat.cluster_centers_
km_288_centroids = km_288_lat.cluster_centers_
km_258_288_centroids = km_258_288_lat.cluster_centers_
km_288_258_centroids = km_288_258_lat.cluster_centers_

km1_order = np.argsort(km_258_centroids[:,0])
km2_order = np.argsort(km_288_centroids[:,0])
km3_order = np.argsort(km_258_288_centroids[:,0])
km4_order = np.argsort(km_288_258_centroids[:,0])

km_258_centroids = km_258_centroids[km1_order,:]
km_288_centroids = km_288_centroids[km2_order,:]
km_258_288_centroids = km_258_288_centroids[km3_order,:]
km_288_258_centroids = km_288_258_centroids[km4_order,:]

km_288_labels = km_288_lat.labels_

km_258_dists = np.linalg.norm(km_258_centroids - km_288_258_centroids,axis=1)
km_288_dists = np.linalg.norm(km_288_centroids - km_258_288_centroids,axis=1)

s=1.5
alpha=0.4
colormap='viridis'
ax = plt.gca()
im = ax.scatter(km_258_dists,km_288_dists,c='r',alpha=alpha,cmap=colormap,s = s)

save_filename=os.path.join(plots_dir,'centroid_dists.pdf')

plt.xlabel('Centroid Dists Model 1')
plt.ylabel('Centroid Dists Model 2')
plt.tight_layout()
plt.axis('square')
plt.savefig(save_filename)
plt.close('all')

## Projecting 258 from 258 model to 258 from 288 model
v1 = np.matmul(np.linalg.pinv(np.matmul(km_258_centroids.T,km_258_centroids)),\
            np.matmul(km_258_centroids.T,km_258_288_centroids))
# Projecting 288 from 258 model to 288 from 288 model
v2 = np.matmul(np.linalg.pinv(np.matmul(km_288_258_centroids.T,km_288_258_centroids)),\
            np.matmul(km_288_258_centroids.T,km_288_centroids))

v3 = np.matmul(np.linalg.pinv(np.matmul(km_258_centroids.T,km_258_centroids)),\
            np.matmul(km_258_centroids.T,km_288_centroids))


proj_288_w_258 = np.matmul(km_288_258_centroids,v1)
proj_258_w_288 = np.matmul(km_258_centroids,v2)
orig_proj_258 = np.matmul(km_258_centroids,v1)
orig_proj_288 = np.matmul(km_288_258_centroids,v2)

new_proj = np.matmul(km_288_258_centroids,v3)

model = VAE(save_dir = 'new_model')
model.load_state(model_288_fname)
with torch.no_grad():
    proj_to_288_288 = model.decode(torch.from_numpy(proj_288_w_258).float())
    thru_288_288 = model.decode(torch.from_numpy(km_288_centroids).float())

    proj_to_288_258 = model.decode(torch.from_numpy(proj_258_w_288).float())
    thru_288_258 = model.decode(torch.from_numpy(km_258_288_centroids).float())

    orig_proj_to_288_258 = model.decode(torch.from_numpy(orig_proj_258).float())
    orig_proj_to_288_288 = model.decode(torch.from_numpy(orig_proj_288).float())

    orig_258_centroids = model.decode(torch.from_numpy(km_258_centroids).float())
    new_proj = model.decode(torch.from_numpy(new_proj).float())

proj_to_288_288 = proj_to_288_288.view(-1,128,128).cpu().detach().numpy()
thru_288_288 = thru_288_288.view(-1,128,128).cpu().detach().numpy()
orig_proj_to_288_288 = orig_proj_to_288_288.view(-1,128,128).cpu().detach().numpy()

proj_to_288_258 = proj_to_288_258.view(-1,128,128).cpu().detach().numpy()
thru_288_258 = thru_288_258.view(-1,128,128).cpu().detach().numpy()
orig_proj_to_288_258 = orig_proj_to_288_258.view(-1,128,128).cpu().detach().numpy()

orig_258_centroids = orig_258_centroids.view(-1,128,128).cpu().detach().numpy()
new_proj = new_proj.view(-1,128,128).cpu().detach().numpy()

all_specs_288 = np.stack([orig_proj_to_288_288,proj_to_288_288,thru_288_288])
all_new_specs = np.stack([orig_258_centroids,new_proj,thru_288_288])
all_specs_258 = np.stack([orig_proj_to_288_258,proj_to_288_258,thru_288_258])
save_filename1=os.path.join(plots_dir,'reconstructions_288.pdf')
save_filename2=os.path.join(plots_dir,'reconstructions_258.pdf')
save_filename3=os.path.join(plots_dir,'reconstructions_new.pdf')

grid_plot(all_specs_288,gap=(2,6),filename=save_filename1)
grid_plot(all_specs_258,gap=(2,6),filename=save_filename2)
grid_plot(all_new_specs,gap=(2,6),filename=save_filename3)

#all_data = np.vstack((km_288_centroids,proj_288_w_258))
#data_inds_258_288 = np.hstack((np.ones((np.shape(km_288_centroids)[0]),dtype=bool),\
#                np.zeros((np.shape(proj_288_w_258)[0]), dtype=bool)))
#data_inds_proj_258 = np.hstack((np.zeros((np.shape(km_288_centroids)[0]),dtype=bool),\
#                np.ones((np.shape(proj_288_w_258)[0]), dtype=bool)))

#s=1.5
#alpha=0.4
#colormap='viridis'

#save_filename=os.path.join(plots_dir,'umap_projs.pdf')
#fig,axs = plt.subplots(2,8)
#for group in range(2):
#    if group == 0:
#        data = proj_to_288_288
#    else:
#        data = thru_288_288
#    for img in range(8):
#        axs[group,img].plot(np.squeeze(data[img,:,:]))
#        axs[group.img].axis('off')
#        if img == 0 & group == 0:
#            axs[group,img].set_title('Bird Centroids Projected to Model 2')
#        elif img == 0 & group == 1:
#            axs[group,img].set_title('Bird Centroids Through Model 2')

#im1 = ax.scatter(umap_258_288[:,0],umap_258_288[:,1],c='r',alpha=alpha,cmap=colormap,s = s)
#im2 = ax.scatter(umap_proj_258[:,0],umap_proj_258[:,1],c='b',alpha=alpha,cmap=colormap,s=s)

#plt.legend(['Centroids: Bird 1 Thru Mod 2', 'Centroids: Bird 1 proj to Bird 2'])

#plt.xlabel('UMAP component 1')
#plt.ylabel('UMAP component 2')
#plt.tight_layout()
#plt.xlabel('umap dim 1')
#plt.ylabel('umap dim 2')


#print(data_inds_258_288)
#print(data_inds_258_288.shape)


######## DONT USE THIS CMON NOW #################
#transform = umap.UMAP(n_components=2, n_neighbors=2, min_dist=0.001, \
    #metric='euclidean', random_state=42)

#umap_all_dat = transform.fit_transform(all_data)
#print(umap_all_dat.shape)

#umap_258_288 = umap_all_dat[data_inds_258_288,:]
#umap_proj_258 = umap_all_dat[data_inds_proj_258]


#save_filename=os.path.join(plots_dir,'cluster_inertia.pdf')
#ax = plt.gca()

#color_list = ['b','r']
#x_ax = list(range(1,15))
#plt.plot(x_ax, td_258_all, c='b')
#plt.plot(x_ax, td_288_all, c='r')
#plt.tight_layout()
#plt.xlabel('n clusters')
#plt.ylabel('inertia')
#plt.savefig(save_filename)
#plt.close('all')

#model_258_dc.clear_projections()
#model_288_dc.clear_projections()

#latent_projection_plot_DC(model_258_dc, alpha=0.25, s=0.5,filename=os.path.join(plots_dir,'latents_umap_258.pdf'))
#latent_projection_plot_DC(model_288_dc, alpha=0.25, s=0.5,filename=os.path.join(plots_dir,'latents_umap_288.pdf'))
