from model import *
from hyperparameters import args

trainer = pl.Trainer(max_epochs=20,progress_bar_refresh_rate=20, auto_scale_batch_size=True) # bach size auto finder
tuner = tuner.tuning.Tuner(trainer)

if args.mode == 'gmlp':
    print('Traing using gMLP...')
    trainer.fit(gmlp,train_loader)
elif args.mode == 'basic-mlp':
    print('Traing using basic-MLP...')
    trainer.fit(basic_mlp, train_loader)
    print('finished')