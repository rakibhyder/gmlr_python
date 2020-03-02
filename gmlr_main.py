import numpy as np
from os.path import isdir
from os import mkdir
from gmlr import gmlr
import matplotlib.pyplot as plt
import time

test_dir='/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/cvpr_2019/Pushing_the_limit_experiments/data'
matrix_path='/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/cvpr_2019/Pushing_the_limit_experiments/new_exp/measurement_matrix'

ngf=64
vid_dir='/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/cvpr_2019/Pushing_the_limit_experiments/new_exp'+'/'+str(ngf)+'_'+str(ngf)
result_dir='/media/rakib/Data/Research/GenerativeModel/Codes/GLO/My/pytorch/cvpr_2019/Pushing_the_limit_experiments/new_exp/result/'
if not isdir(result_dir):
    mkdir(result_dir)

test_dataset='rot_mnist'
print(test_dataset)
measurement_type='missing'

if measurement_type=='missing':
    measurement_pool=[0.8]
elif measurement_type=='compressive':
    measurement_pool=[29]

noise_add=False
noise_snr=20
if measurement_type=='denoising':
    noise_add=True
    noise_snr=20
    measurement_rate=0
     
update_type='both'
lamda=1
low_rank_pool=[2]#range(32)
test_epochs=1999
lr_decay=500

seed_pool=[100] #,200,300,400,500,600,700,800,900,1000
run=len(seed_pool)



for m in measurement_pool:
    measurement_rate=m
    for low_rank in low_rank_pool:    
        for i in range (0,run):
            seed=seed_pool[i]
            start_time = time.time()
            psnr, x_rec,x_test,y, trained_net,z_test,meas_loss,rec_loss=gmlr(test_dataset,test_dir=test_dir,ngf=ngf,measurement_type=measurement_type, measurement_rate=measurement_rate,
                 matrix_path=matrix_path,seed=seed,update_type=update_type,test_epochs=test_epochs,lr_decay=lr_decay,noise_add=noise_add,noise_snr=noise_snr, low_rank=low_rank,lamda=lamda,save_vid=True,video_dir=vid_dir)
            print(psnr)
            if measurement_type=='denoising':
                res_file=result_dir+'/'+test_dataset+'_'+str(ngf)+'_'+str(ngf)+'_'+measurement_type+str(noise_snr)+'_'+update_type+'_lamda_'+str(lamda)+'_rank_'+str(low_rank)+'.txt'
            else:                    
                res_file=result_dir+'/'+test_dataset+'_'+str(ngf)+'_'+str(ngf)+'_'+measurement_type+str(measurement_rate)+'_'+update_type+'_lamda_'+str(lamda)+'_rank_'+str(low_rank)+'.txt'
            f = open(res_file, "a+")
            f.write(str(psnr)+'\n')
            f.close()
            
            plt.figure()
             
            plt.plot(np.log10(meas_loss/4))
            plt.hold(True)
            plt.xlabel('Epochs')
            plt.ylabel('Measurement loss (log scaled) during testing')
            if measurement_type=='denoising':
                plt.title(measurement_type+str(noise_snr))
                plt.savefig(vid_dir+'/'+test_dataset+'_meas_'+update_type+'_'+measurement_type+str(noise_snr)+'_lamda_'+str(lamda)+'_rank_'+str(low_rank)+'_seed_'+str(seed)+'_'+str(psnr)+'dB.jpg')
                plt.show()
            else:
                plt.title(measurement_type+str(measurement_rate)) 
                plt.savefig(vid_dir+'/'+test_dataset+'_meas_'+update_type+'_'+measurement_type+str(measurement_rate)+'_lamda_'+str(lamda)+'_rank_'+str(low_rank)+'_seed_'+str(seed)+'_'+str(psnr)+'dB.jpg')
                plt.show()

            plt.show()
            
            plt.figure()
             
            plt.plot(np.log10(rec_loss/4))
            plt.hold(True)
            plt.xlabel('Epochs')
            plt.ylabel('Reconstruction loss (log scaled) during testing')
            if measurement_type=='denoising':
                plt.title(measurement_type+str(noise_snr))                
                plt.savefig(vid_dir+'/'+test_dataset+'_rec_'+update_type+'_'+measurement_type+str(noise_snr)+'_lamda_'+str(lamda)+'_rank_'+str(low_rank)+'_seed_'+str(seed)+'_'+str(psnr)+'dB.jpg')
                plt.show()
            else:
                plt.title(measurement_type+str(measurement_rate))                   
                plt.savefig(vid_dir+'/'+test_dataset+'_rec_'+update_type+'_'+measurement_type+str(measurement_rate)+'_lamda_'+str(lamda)+'_rank_'+str(low_rank)+'_seed_'+str(seed)+'_'+str(psnr)+'dB.jpg')
                plt.show()
                        
            end_time = time.time()
            print('Time taken:',end_time-start_time)



from scipy import linalg
import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)
sequence_size=32
zt=z_test
rank=2
u, s, vh = np.linalg.svd(zt, full_matrices=False)
zt=np.dot(u * np.append(s[0:rank],np.zeros(len(s)-rank)), vh)

zb=linalg.orth(zt.T)
print(np.linalg.matrix_rank(zb))
zb=zb.T
a1=np.matmul(zt[0:sequence_size,:],np.linalg.pinv(zb))
#a2=np.matmul(zt[sequence_size:ln,:],np.linalg.pinv(zb))

area=50

plt.figure(figsize=(10, 10))
start_seq=0
end_seq=sequence_size 
ax=plt.subplot()
plt.scatter(a1[start_seq:end_seq,0], a1[start_seq:end_seq,1],s=area,alpha=1)
plt.grid(True)
plt.xlabel('Weight on orthogonal basis 1',fontsize=20)
plt.ylabel('Weight on orthogonal basis 2', fontsize=20)
 
#plt.title('Weights for representing each z of the sequence (rank=2)')
for i in range(a1.shape[0]):
    ax.annotate('  '+str(i+1), (a1[i,0], a1[i,1]-0.1), fontsize=18)  