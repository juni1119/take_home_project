import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="/home/yuhu/experiments/Dehaze_NH_221024_163156/checkpoint/I80000_E1455", help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val') 
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    # command에서 받아온 argument를 처리한다. inference 단계에서는 --config만 들어온다.

    args = parser.parse_args() # command "python infer.py --config $optpath" 에서 opt 정보가 포함된 json 파일을 args가 받아온다.
    opt = Logger.parse(args) # args.config로부터 opt_path정보를 받아 json 파일을 읽고 opt에 dict를 저장한다. 
    print(opt['path'])

    opt = Logger.dict_to_nonedict(opt) # opt가 dict인 경우 하위 dict 모두 포함해서 NoneDict로 바꿔서 return한다

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    # train phase의 경우 "25-05-15 16:43:12.237 - INFO : Training started" 형식의 로그를
    # 1) FileHandler를 통해 (root)/(phase).log 파일에 저장하고,
    # 2) StreamHandler를 통해 (screen = true) stdout에 출력한다.
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    # val phase의 경우는 StreamHandler를 add하지 않는다 (screen = false)

    logger = logging.getLogger('base')
    # 'base' 로거를 생성한다 
    logger.info(Logger.dict2str(opt))
    # opt를 str형식으로 펼쳐서 info 수준으로 출력한다.
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
    # TensorBoard에서 모니터링할 수 있도록 로그기록기를 생성한다.

    # opt에 따라 "Weights&Biases"에 학습 로그를 업로드할 수 있도록 객체를 생성해준다.
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # infer 단계에서는 phase가 val인 dataset만 고려해준다
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            # 주어진 opt와 phase에 맞게 LRHRDataset인 object를 생성한다 
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
            # torch.utils.data.DataLoader를 활용한다.
            # -> __len__()과 __getitem__() 함수를 갖는 dataset class에 대해 (LRHRDataset)
            #    batch size만큼 getitem하여 반환하고
            #    연속해서 그 다음 batch도 반환하여 iterator 역할을 한다
            #    val phase의 경우 batch 1개에 대해 shuffle 없이 반환한다 
    logger.info('Initial Dataset Finished')

    diffusion = Model.create_model(opt)
    # opt에 맞게 DDPM object를 만들어서 'diffusion'에 할당한다
    # 그 과정에서 self.netG가 생성된다. <- define_G(opt)
    # -> opt['model']의 정보에 따라 
    #    model = unet.UNet( opts )
    #    netG = diffusion.GaussianDiffusion( model, opts )
    # 그리고 opt에 맞게 'cuda'에서 동작할지, 'cpu'에서 동작할지 결정한다.
    logger.info('Initial Model Finished')
 
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    # 생성한 DDPM object의 netG의 self.sqrt_alphas_cumprod_prev를 주어진 opt의 beta_schedule에 맞게 설정한다

    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    # 생성된 이미지를 저장할 디렉토리를 opt에 지정된 경로에 만들어준다 

    avg_psnr = 0.0
    for _,  val_data in enumerate(val_loader):
        idx += 1


        diffusion.feed_data(val_data)
        # diffusion의 self.data를 val_data로 설정한다.(set_device 포함)
        diffusion.test(continous=True)
        # diffusion의 self.data를 이용해 self.SR, 즉 최종 output(stage 1 - stage2를 모두 거친)을 만들어낸다. 
        
        
        visuals = diffusion.get_current_visuals()
        # diffusion의 self.SR(Out)과 self.data에 포함된 hazy-image(LR), gt(HR)이 담긴 ordered dict를 visuals에 할당한다 
        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
        # 시각화를 위해 RGB 이미지로 변환한다. (uint8 : [0, 255])
        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            sr_img = visuals['Out']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_out_{}.png'.format(result_path, current_step, idx, iter))
        else:
          # grid 모드 !
            sr_img = Metrics.tensor2img(visuals['Out'])  # uint8
            # output 이미지도 RGB 이미지로 변환한다 
            Metrics.save_img(Metrics.tensor2img(visuals['Out'][-1]), '{}/{}_{}_out.png'.format(result_path, current_step, idx))
            # 복원된 이미지 중 마지막 이미지만 저장한다. {result_path}/{current_step}_{idx}_out.png 형식으로 
            # 마지막 이미지를 저장하는 이유는 self.SR이 최종 이미지 x0말고도 중간 단계의 xt를 포함하는 경우를 고려했기 때문 

        Metrics.save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
        # input image (hazy-image, LR)도 따로 저장해준다.

        avg_psnr += Metrics.calculate_psnr(Metrics.tensor2img(visuals['Out'][-1]), hr_img)
        # calculate_psnr : 복원 이미지와 gt 사이의 유사성을 계산 

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(lr_img, Metrics.tensor2img(visuals['Out'][-1]), hr_img)
            # opt['log_infer']가 true일 경우 복원과정을 WandB에 기록 

    avg_psnr = avg_psnr / idx
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
    # 다시 train 모드의 schedule로 변경 
    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    # 최종적으로 idx(총 배치수)를 나눠 계산한 PSNR 출력 

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
        # 최종적으로 log_eval_data에서 로그한 이미지들을 table로 WandB에 올린다 
