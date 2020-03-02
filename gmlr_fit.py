import torch
import numpy as np
import copy



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def gmlr_fit(generator,
             y,
             M,nc,ngf,ndf,nz,
             measurement_type,seed,
             test_epochs,
             update_type,
             x_org=[],
             single=False,
             pretrained=False,
             pretrained_weight=None,
             gamma_lr=0.00025,
             z_lr=10,
             lr_decay=4000,
             decay_rate=0.75,
             gamma_opt='adam',
             z_opt='sgd',
             z_init=None,
             lamda=1,
             low_rank=0,
             low_rank_type='svd',
             find_best=True
             ):
             
    test_size=y.shape[0]
    sequence_size=test_size
    test_batch_size=test_size
    
    test_alpha=np.float(z_lr)*test_size
    
    best_margin=0.99
    best_mse=100000 # initial value
    
    if not z_init==None:
        z_test=z_init
    else:
#        np.random.seed(seed)
        z_test=np.random.normal(loc=0, scale=1.0, size=(test_size,nz))
        for i in range(z_test.shape[0]):
            z_test[i,:] = z_test[i, :] / np.linalg.norm(z_test[i, :], 2)
            
            

    if single:
        test_alpha=z_lr
        test_batch_size=1
        
    batch_no=np.int(np.ceil(test_size/np.float(test_batch_size)))
    idx=np.arange(test_size)
    loss_test=[]   
    loss_rec=[]
    x_rec=np.zeros((test_size,nc,ngf,ndf))
    
    for batch_idx in range(0,batch_no):
        if pretrained:
            generator=torch.load( pretrained_weight)
        if gamma_opt=='sgd':
            optimizer = torch.optim.SGD(generator.parameters(), gamma_lr)
        elif gamma_opt=='adam':
            optimizer = torch.optim.Adam(generator.parameters(), gamma_lr)
        elif gamma_opt=='adadelta':
            optimizer = torch.optim.Adadelta(generator.parameters(), gamma_lr) 
#                summary(generator,(nz,1,1))
        loss_epoch=[]
        loss_rec_epoch=[]
        epoch_idx=idx
    #        np.random.shuffle(epoch_idx)
#            if epoch%100==0:
#                print(epoch)
        if measurement_type=='denoising':
            x_batch=y[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:,:,:]
            x_batch_tensor=torch.cuda.FloatTensor(x_batch).view(-1,nc,ngf,ndf)
        elif measurement_type=='missing':
            x_batch=y[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:,:,:]
            x_batch_tensor=torch.cuda.FloatTensor(x_batch).view(-1,nc,ngf,ndf)
            mask_tensor=torch.cuda.FloatTensor(M[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:,:]).view(-1,nc,ngf,ndf)
        elif measurement_type=='linear':
            x_batch=y[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:,0]
            x_batch_tensor=torch.cuda.FloatTensor(x_batch).view(-1, x_batch.shape[1],1)
            mask_tensor=torch.cuda.FloatTensor(M[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:,:]).view(-1,M.shape[1],nc*ngf*ndf)
        elif measurement_type=='compressive':
            x_batch=y[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:,:,:]
            x_batch_tensor=torch.cuda.FloatTensor(x_batch).view(-1,nc,M.shape[2],M.shape[2])
            
            mask_tensor=torch.cuda.FloatTensor(M[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:,:]).view(-1,nc,M.shape[2],ngf)

        for epoch in range (0,test_epochs):
            if (epoch+1)%lr_decay==0:
                print(epoch)
                gamma_lr=gamma_lr*decay_rate
                test_alpha=test_alpha
                for param_group in optimizer.param_groups:
                    param_group['lr'] = gamma_lr
#                if gamma_opt=='sgd':
#                    optimizer = torch.optim.SGD(generator.parameters(), gamma_lr)
#                elif gamma_opt=='adam':
#                    optimizer = torch.optim.Adam(generator.parameters(), gamma_lr)
#                elif gamma_opt=='adadelta':
#                    optimizer = torch.optim.Adadelta(generator.parameters(), gamma_lr)     
                

            z_batch=z_test[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:]
            z_batch_tensor=torch.autograd.Variable(torch.cuda.FloatTensor(z_batch).view(-1, nz, 1, 1),requires_grad=True)

            x_hat = generator(z_batch_tensor)
            if measurement_type=='linear':
                x_hat_mask=torch.matmul(mask_tensor,x_hat.view(-1, nc* ngf*ndf,1))
            elif measurement_type=='denoising':
                x_hat_mask=x_hat
            elif measurement_type=='missing':
                x_hat_2=x_hat/2+0.5
                x_hat_mask_1=torch.mul(x_hat_2,mask_tensor)
                x_hat_mask=2*x_hat_mask_1-1
            elif measurement_type=='compressive':
#                print(M.shape)
#                print(mask_tensor.shape)
#                print(torch.transpose(mask_tensor,2,3).shape)
                x_hat_mask=torch.matmul(torch.matmul(mask_tensor,x_hat),torch.transpose(mask_tensor,2,3))

                
            loss_mse=(x_hat_mask - x_batch_tensor).pow(2).mean()
            loss_epoch.append(loss_mse.item())
            
            if find_best:
                if loss_mse<best_margin*best_mse:
                    best_net=copy.deepcopy(generator)
                    best_z=z_test
                    best_mse=loss_mse
            
#                    z_for=z_batch_tensor[1:test_batch_size,:]
#                    z_back=z_batch_tensor[0:test_batch_size-1,:]
#                    loss_z=(z_for-z_back).pow(2).mean()
            if test_size>2 and (not single) and (not lamda==1):
                for i in range (0, np.int(np.ceil(z_batch.shape[0]/np.float(sequence_size)))):
                    z_for=z_batch_tensor[i*sequence_size+1:np.min([(i+1)*sequence_size,test_batch_size]),:]
                    z_back=z_batch_tensor[i*sequence_size:np.min([(i+1)*sequence_size,test_batch_size])-1,:]
                    if i==0:
                        loss_z=(z_for-z_back).pow(2).mean()
                    else:
                        loss_z=loss_z+(z_for-z_back).pow(2).mean()
                loss_z=loss_z
#                loss_z=torch.abs((z_for-z_back)).mean()
                loss=lamda*loss_mse+(1-lamda)*loss_z                    
            else:
                loss=loss_mse

            optimizer.zero_grad()
            loss.backward(retain_graph=True)   

            with torch.no_grad():        
                    
                if update_type=='z' or update_type=='both':
                    z_grad = z_batch_tensor.grad.data.cuda()
                    if z_opt=='sgd':
                        z_update = z_batch_tensor - test_alpha * z_grad
    
                    z_update = z_update.cpu().detach().numpy()
                    z_update=np.reshape(z_update,z_batch.shape)
#                        if z_norm_type=='unit_norm':
#                            for i in range(z_update.shape[0]):
#                                z_update[i,:] = z_update[i, :] / np.linalg.norm(z_update[i, :], 2)
                    if low_rank!=0:
                        if low_rank_type=='svd':
                            u, s, vh = np.linalg.svd(z_update, full_matrices=False)
                            z_update=np.dot(u * np.append(s[0:low_rank],np.zeros(len(s)-low_rank)), vh)
                        elif low_rank_type=='pca':
                            z_mean=np.mean(z_update,axis=0)
                            z_temp=z_update-z_mean
                            u, s, vh = np.linalg.svd(z_temp, full_matrices=False)
                            z_new=np.dot(u * np.append(s[0:low_rank-1],np.zeros(len(s)-low_rank+1)), vh)
                            z_update=z_new+z_mean

                    z_test[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:]=z_update
    
                    z_update_tensor=torch.autograd.Variable(torch.cuda.FloatTensor(z_update).view(-1, nz, 1, 1))
                    x_hat = generator(z_update_tensor)
            x_rec[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:,:,:]=np.reshape(x_hat.cpu().detach().numpy(),(x_batch.shape[0],nc,ngf,ndf))
            
            if x_org.size:
                loss_rec_epoch.append(np.mean((x_rec-x_org)**2))
                
            if update_type=='gen' or update_type=='both':
                optimizer.zero_grad()
                loss.backward()   
                optimizer.step()
        
        loss_rec.append(np.array(loss_rec_epoch))
        loss_test .append(np.array(loss_epoch))

#            np.savez('/media/rakib/Data/Data/GLO/Results/'+train_dataset+'_'+test_z_type+'_'+str(nz)+'_alpha'+str(test_alpha)+'_epochs'+str(test_epochs)+'_'+str(test_batch_size)+subtype+'_subsample_'+str(sub_dim),z_test=z_test,mask=mask)


        if find_best:
            z_test=best_z
            generator=copy.deepcopy(best_net)
            best_z_tensor=torch.autograd.Variable(torch.cuda.FloatTensor(best_z).view(-1, nz, 1, 1))
            x_hat = best_net(best_z_tensor)
            x_rec[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:,:,:]=np.reshape(x_hat.cpu().detach().numpy(),(x_batch.shape[0],nc,ngf,ndf))
        
    meas_loss=np.mean(np.array(loss_test),axis=0)
    if x_org.size:
        rec_loss=np.mean(np.array(loss_rec),axis=0)
    else:
        rec_loss=None
    
    return x_rec, generator,z_test,meas_loss, rec_loss