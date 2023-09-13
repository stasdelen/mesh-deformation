import torch
from remin import callbacks
from remin.residual import make_loader
from remin.solver import Solver, make_trainer
from remin.solver.residual_loss import EagerLoss
from torch import nn


class SoftMesh(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 3)
        )
    
    def forward(self, x, y, z):
        Uin = torch.hstack((x, y, z))
        U = self.linear_tanh_stack(Uin)
        return U[:,0:1], U[:,1:2], U[:,2:3]
    
    def calc(self, U):
        return self.linear_tanh_stack(U)
    
def trainSoftModel(residuals, eval_vertices, name):
	softModel = SoftMesh()

	loader = make_loader(
		residuals,
		fully_loaded=True
	)

	epochs = 3#0000
	lr = 1e-3
	gamma = 0.9

	optimizer = torch.optim.Adam(softModel.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=gamma)
	resloss = EagerLoss(nn.MSELoss())
	trainer = make_trainer(loader,
						   optimizer=optimizer,
						   scheduler=scheduler,
						   residual_loss=resloss)
	
	solver = Solver(softModel,
					name=name,
					save_folder='./outputs/' + name,
					trainer=trainer)
	
	solver.reset_callbacks(
		callbacks.TotalTimeCallback(),
		callbacks.SaveCallback(),
		callbacks.LogCallback(log_epoch=1000, log_progress=5),
        callbacks.ToleranceCallback(tolerance=2e-3),
		callbacks.PlotCallback(state='residual', name='ressloss.png')
	)

	solver.fit(epochs)
	return softModel.calc(eval_vertices).cpu().detach().numpy()