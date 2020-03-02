from glob import glob
from scipy import misc
from os.path import isdir
from os import mkdir
import numpy as np
#import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import skvideo.io
#from torchsummary import summary
from gmlr_dataprep import gmlr_dataprep
from gmlr_net import gmlr_net
from gmlr_fit import gmlr_fit


def gmlr(test_dataset,
         test_dir,ngf,
         measurement_type,
         measurement_rate,
         matrix_path,
         seed,
         update_type,test_epochs,lr_decay,
         low_rank=0,lamda=1,noise_add=False,noise_snr=30,
         save_vid=True,
         video_dir=None):
    
    ndf=ngf
    
    
    ll=sorted(glob(test_dir+'/'+test_dataset+'/seq1'+'/*.jpg'))
#    ll=sorted(glob('/media/rakib/Data/Data/KTH/Frames/handwaving/person04_handwaving_d4_uncomp/*.jpg'))
    
    ll=ll[0:32]
    ln=len(ll)
    if ln==0:
        print('No image found')
#    print(ln)
    x_test=[]
    for i in range (0,ln):
        img=misc.imread(ll[i])
        if img.ndim==2:
            nc=1
        else:
            nc=3
        w=img.shape[0]
        h=img.shape[1]
        crop=np.min([h,w])
        if nc==3:
            img1=img[(np.int(w-crop)/2):(np.int(w+crop)/2),(np.int(h-crop)/2):(np.int(h+crop)/2),:]
        elif nc==1:
            img1=img[(np.int(w-crop)/2):(np.int(w+crop)/2),(np.int(h-crop)/2):(np.int(h+crop)/2)]
        
        img=misc.imresize(img1,[ngf,ndf])
        if measurement_type=='linear':
            img=rgb2gray(img)
            nc=1
        img=img/255.0
        temp=np.zeros((nc,ngf,ndf))
        if nc==3:
            for chan in range (0,nc):
                temp[chan,:,:]=img[:,:,chan]
        elif nc==1:
            temp[0,:,:]=img[:,:]    
        x_test.append(temp)
    x_test=np.array(x_test)
    

    if ngf==32:
        nz=32
        ng=64
        test_epochs=1000
        gamma_lr=0.00025
    elif ngf==64 and nc==1:
        nz=64
        ng=64
        test_epochs=1000
        gamma_lr=0.00025
    elif ngf==64 and nc==3:
        nz=256
        ng=64
        test_epochs=1000
        gamma_lr=0.00025
    elif ngf==256:
        nz=512
        ng=64
        test_epochs=4000
        gamma_lr=0.0025
    elif ngf==512:
        nz=512
        ng=128  
        test_epochs=16000
        gamma_lr=0.0025
        
    generator=gmlr_net(nz,ngf,ndf,nc,ng,seed)
#    summary(generator,(nz,1,1))
    x_org, y, M=gmlr_dataprep(x_test,measurement_type,measurement_rate,matrix_path,seed,
              same_matrix=False,make_new_mtx=False,noise_add=noise_add,noise_type='Gaussian', noise_mean=0,noise_std=1,noise_snr=noise_snr)
    
    
    x_rec, trained_net,z_test,meas_loss, rec_loss=gmlr_fit(generator=generator,y=y,x_org=x_org,M=M,nc=nc,ngf=ngf,ndf=ndf,nz=nz,measurement_type=measurement_type,seed=seed,test_epochs=test_epochs, 
            update_type=update_type,single=False, pretrained=False, pretrained_weight=None, gamma_lr=gamma_lr,z_lr=10,lr_decay=lr_decay,decay_rate=0.75,
             gamma_opt='adam',z_opt='sgd',z_init=None, lamda=lamda,low_rank=low_rank,low_rank_type='pca',find_best=True)
    
    x_rec=x_rec/2+0.5
    mse=np.mean((x_rec-x_test)**2)
    psnr=20*np.log10((np.max(x_test)-np.min(x_test))/np.sqrt(mse))
    
    if save_vid:
        if not isdir(video_dir):
            mkdir(video_dir)
        
        outputdata = x_test* 255
        outputdata = outputdata.astype(np.uint8)
        x_test_vid=np.zeros((outputdata.shape[0],ngf, ndf,nc))
        for i in range(0,outputdata.shape[0]):
            temp=outputdata[i]
            temp1=np.zeros((ngf, ndf,nc))
            for chan in range (0,nc):
                temp1[:,:,chan]=temp[chan,:,:]
            x_test_vid[i]=temp1
        writer = skvideo.io.FFmpegWriter(video_dir+'/'+test_dataset+'_resize'+'_'+str(ngf)+'_'+str(ndf)+'.mp4')
        for i in xrange(x_test_vid.shape[0]):
                writer.writeFrame(x_test_vid[i, :, :, :])
        writer.close()
    
        outputdata = x_rec * 255
        outputdata = outputdata.astype(np.uint8)
        x_test_vid=np.zeros((outputdata.shape[0],ngf, ndf,nc))
        for i in range(0,outputdata.shape[0]):
            temp=outputdata[i]
            temp1=np.zeros((ngf, ndf,nc))
            for chan in range (0,nc):
                temp1[:,:,chan]=temp[chan,:,:]
            x_test_vid[i]=temp1
        if measurement_type=='denoising':
            writer = skvideo.io.FFmpegWriter(video_dir+'/'+test_dataset+'_rec_'+update_type+'_'+measurement_type+str(noise_snr)+'_lamda_'+str(lamda)+'_rank_'+str(low_rank)+'_seed_'+str(seed)+'_'+str(psnr)+'dB'+'.mp4')
        else:
            writer = skvideo.io.FFmpegWriter(video_dir+'/'+test_dataset+'_rec_'+update_type+'_'+measurement_type+str(measurement_rate)+'_lamda_'+str(lamda)+'_rank_'+str(low_rank)+'_seed_'+str(seed)+'_'+str(psnr)+'dB'+'.mp4')
        for i in xrange(x_test_vid.shape[0]):
                writer.writeFrame(x_test_vid[i, :, :, :])
        writer.close()
        

        if measurement_type=='missing' or (measurement_type=='denoising' and noise_add==True) or measurement_type=='compressive':
            if (measurement_type=='denoising' and noise_add==True):                
                outputdata = (y/np.max([2,(np.max(y)-np.min(y))])+0.5) * 255
            elif measurement_type=='missing':
                outputdata = (y/2+0.5) * 255
            elif measurement_type=='compressive':
                outputdata = (y/(np.max(y)-np.min(y))+0.5) * 255
                
            outputdata = outputdata.astype(np.uint8)
            if measurement_type=='compressive':
                x_test_vid=np.zeros((outputdata.shape[0],measurement_rate, measurement_rate,nc))
            else:                    
                x_test_vid=np.zeros((outputdata.shape[0],ngf, ndf,nc))
            for i in range(0,outputdata.shape[0]):
                temp=outputdata[i]
                
                temp1=np.zeros((x_test_vid.shape[1],x_test_vid.shape[2],nc))
                for chan in range (0,nc):
                    temp1[:,:,chan]=temp[chan,:,:]
                x_test_vid[i]=temp1
            if measurement_type=='denoising':
                writer = skvideo.io.FFmpegWriter(video_dir+'/'+test_dataset+'_input_'+'_'+measurement_type+str(noise_snr)+'_seed_'+str(seed)+'.mp4')
            else:                    
                writer = skvideo.io.FFmpegWriter(video_dir+'/'+test_dataset+'_input_'+'_'+measurement_type+str(measurement_rate)+'_seed_'+str(seed)+'.mp4')
            for i in xrange(x_test_vid.shape[0]):
                    writer.writeFrame(x_test_vid[i, :, :, :])
            writer.close()
            
    return psnr, x_rec,x_test,y, trained_net,z_test,meas_loss, rec_loss