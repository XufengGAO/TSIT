import sys
from collections import OrderedDict

import data
from options.train_options import TrainOptions
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
# parse options
opt = TrainOptions().parse()
# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))
dataset_size = len(dataloader) 

# create tool for visualization
visualizer = Visualizer(opt)
# SummaryWriter instance
tb_folder = os.path.join("/home/xugao/gitRepo/TSIT/runs", opt.name)
if os.path.exists(tb_folder) is False:
    os.makedirs(tb_folder)
tb_writer = SummaryWriter(log_dir=tb_folder)

for epoch in tqdm(iter_counter.training_epochs()):
    iter_counter.record_epoch_start(epoch)
    visualizer.reset()              
    for i, data_i in enumerate(tqdm(dataloader), start=iter_counter.epoch_iter):  # set start index of enumerate
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_losses(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
        #     print(losses)
        #     visualizer.plot_current_losses(epoch, float(iter_counter.epoch_iter) / dataset_size, losses)
        #     for k, v in losses.items():
        #         if k in ['D_Fake', 'D_real']:
        #             tb_writer.add_scalars('D Loss', {k:float(format(v.mean().float(), '.3f'))}, epoch + float(iter_counter.epoch_iter) / dataset_size)
        #         if k in ['KLD', 'GAN', 'GAN_Feat', 'VGG']:
        #             tb_writer.add_scalars('G Loss', {k:float(format(v.mean().float(), '.3f'))}, epoch + float(iter_counter.epoch_iter) / dataset_size)

        if iter_counter.needs_displaying():
            if opt.task == 'SIS':
                visuals = OrderedDict([('input_label', data_i['label'][0]),
                                       ('synthesized_image', trainer.get_latest_generated()[0]),
                                       ('real_image', data_i['image'][0])])
            else:
                visuals = OrderedDict([('content', data_i['label'][0]),
                                       ('synthesized_image', trainer.get_latest_generated()[0]),
                                       ('style', data_i['image'][0])])
 
            save_result = iter_counter.total_steps_so_far % opt.update_html_freq == 0
            visualizer.display_current_results(visuals, epoch, save_result)
            
        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
