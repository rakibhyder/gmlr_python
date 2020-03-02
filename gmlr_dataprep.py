from os.path import isfile, isdir
from os import mkdir
import numpy as np
import scipy.io as sio

def gmlr_dataprep(data, 
                  measurement_type, 
                  measurement_rate, 
                  matrix_path,
                  seed,
                  same_matrix=False,
                  make_new_mtx=False,
                  noise_add=False, 
                  noise_type='Gaussian',
                  noise_mean=0,
                  noise_std=1,
                  noise_snr=30):
    
    batch_size=data.shape[0]
    ngf=data.shape[2]
    ndf=data.shape[3]    
    nc=data.shape[1] 
    
#    data=data-np.min(data)
#    data=data/np.max(data)
    
    
    #Measurement Matrix
    if measurement_type=='denoising':
        M=np.eye(ngf)  # Dummy
    else:
        if not isdir(matrix_path):
            mkdir(matrix_path)
            
        if not make_new_mtx:
            if not same_matrix:
                if isfile(matrix_path+'/'+str(measurement_type)+'_'+str(measurement_rate)+'_'+str(batch_size)+'_'+str(ngf)+'_'+str(ndf)+'_'+str(nc)+'_'+str(seed)+'.mat'):
                    case_mtx=2
                else:
                    case_mtx=22
                    
            else:
                if isfile(matrix_path+'/'+str(measurement_type)+'_'+str(measurement_rate)+'_'+str(ngf)+'_'+str(ndf)+'_'+str(nc)+'_'+str(seed)+'.mat'):
                    case_mtx=1
                else:
                    case_mtx=11   
        else:
            if not same_matrix:
                case_mtx=22
            else:
                case_mtx=11
                        
        if case_mtx==2:
            Mtx=sio.loadmat(matrix_path+'/'+str(measurement_type)+'_'+str(measurement_rate)+'_'+str(batch_size)+'_'+str(ngf)+'_'+str(ndf)+'_'+str(nc)+'_'+str(seed)+'.mat')
            M=Mtx['M']
        elif case_mtx==1:
            Mtx=sio.loadmat(matrix_path+'/'+str(measurement_type)+'_'+str(measurement_rate)+'_'+str(ngf)+'_'+str(ndf)+'_'+str(nc)+'_'+str(seed)+'.mat')
            M1=Mtx['M1']
            if measurement_type=='missing':
                M= np.zeros((batch_size,nc,ngf,ndf))
                for i in range(0,batch_size):
                    M[i,:,:,:]=M1                
            elif measurement_type=='linear':  
                M= np.zeros((batch_size,measurement_rate,nc*ngf*ndf))
                for i in range(0,batch_size):
                    M[i,:,:]=M1
            elif measurement_type=='compressive':
                M= np.zeros((batch_size,measurement_rate,ngf))
                for i in range(0,batch_size):
                    M[i,:,:]=M1
        elif case_mtx==22 or case_mtx==11:
            np.random.seed(seed)
            if measurement_type=='missing':
                M= np.ones((batch_size,nc,ngf,ndf) ) 
    #            print(M.shape)
                mask_len=np.int(ngf*ndf*measurement_rate)
                idx=np.arange(ngf*ndf)
                np.random.shuffle(idx)
                for i in range (0, batch_size):
                    if case_mtx==22:
                        np.random.shuffle(idx)                
                    M1= np.ones((nc,ngf,ndf))
                    mask_temp =np.ones((1,ngf*ndf))
                    mask_temp[0,idx[0:mask_len]]=0
                    mask_temp=np.reshape(mask_temp,(ngf, ndf)) 
    
                    for j in range (0,nc):
                        M[i,j,:,:]=mask_temp
                        if case_mtx==11:
                            M1[j,:,:]=mask_temp
                
            elif measurement_type=='linear':
                if case_mtx==22:             
                    M=np.random.normal(loc=0.0, scale=1.0/np.sqrt(measurement_rate), size=(batch_size,measurement_rate,nc*ngf*ndf))
                elif case_mtx==11:
                    M=np.zeros((batch_size,measurement_rate,nc*ngf*ndf))
                    M1=np.random.normal(loc=0.0, scale=1.0/np.sqrt(measurement_rate), size=(measurement_rate,nc*ngf*ndf))
                    for i in range (0, batch_size):
                        M[i,:,:]=M1
            elif measurement_type=='compressive':
                M=np.zeros((batch_size,nc,measurement_rate,ngf))
                if case_mtx==22:                                 
                    for i in range (0, batch_size):
                        M1=np.random.normal(loc=0.0, scale=1.0/np.sqrt(measurement_rate), size=(measurement_rate,ngf))
                        for chan in range(0,nc):
                            M[i,chan,:,:]=M1  
                elif case_mtx==11:
                    M1=np.random.normal(loc=0.0, scale=1.0/np.sqrt(measurement_rate), size=(measurement_rate,ngf))
                    for i in range (0, batch_size):
                        for chan in range(0,nc):
                            M[i,chan,:,:]=M1  
                        
            if case_mtx==11:
                sio.savemat(matrix_path+'/'+str(measurement_type)+'_'+str(measurement_rate)+'_'+str(ngf)+'_'+str(ndf)+'_'+str(nc)+'_'+str(seed),{'M1':M1})
            elif case_mtx==22:
                sio.savemat(matrix_path+'/'+str(measurement_type)+'_'+str(measurement_rate)+'_'+str(batch_size)+'_'+str(ngf)+'_'+str(ndf)+'_'+str(nc)+'_'+str(seed),{'M':M})
        

    # Measurements
    if measurement_type=='denoising':
        y=2*data-1
    elif measurement_type=='missing':
        y=np.zeros((batch_size,nc,ngf,ndf))
        for i in range (0, batch_size):
            y[i,:,:,:]=np.multiply(data[i,:,:,:],M[i,:,:,:])
        y=2*y-1
    elif measurement_type=='linear':
        temp=2*data-1
        y=np.zeros((batch_size,measurement_rate,1))
        for i in range (0, batch_size):
            y[i,:,:]=np.matmul(M[i,:,:],temp[i,:,:,:].reshape(1,nc*ngf*ndf,1))
            
    elif measurement_type=='compressive':
        temp=2*data-1
        y=np.zeros((batch_size,nc,measurement_rate,measurement_rate))
        print(M.shape)
        for i in range (0, batch_size):
            for chan in range(0,nc):
                y[i,chan,:,:]=np.matmul(np.matmul(M[i,chan,:,:],temp[i,chan,:,:]),M[i,chan,:,:].T)  
    
    if noise_add:
        if noise_type=='Gaussian': 
             
            y_n=np.zeros(y.shape)   
            np.random.seed(seed)  
            for i in range (0,y.shape[0]):
                sig=y[i]
                noise=np.random.normal(loc=noise_mean, scale=noise_std, size=sig.shape)
#                noise_pow=(np.max(sig)-np.min(sig))*np.sqrt((noise_std**2)*np.mean(sig**2)/(10**(noise_snr/10.0)))
                noise_pow=np.sqrt(np.sum(sig**2)/(np.sum(noise**2)*(10**(noise_snr/10.0))))
                temp=noise*noise_pow
                noisy_sig=sig+temp
                
#                print(10*np.log10(np.sum(sig**2)/(np.sum(temp**2))))
#                print(20*np.log10((np.max(sig)-np.min(sig))/np.sqrt(np.mean((noisy_sig-sig)**2))))
                y_n[i]=noisy_sig
            y=y_n     
                
    return 2*data-1,y, M