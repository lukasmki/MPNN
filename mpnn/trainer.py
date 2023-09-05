import os
import time
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from mpnn.data import make_batches


class Trainer:
    def __init__(self, device, model, loss_fn, optimizer, output_path, lr_scheduler):
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.multi_gpu = (type(device) is list) and (len(device) > 1)
        self.parallelized = False

        self.path_iter = 1
        self.epoch = 0
        self.best_val_loss = float("inf")

        # subdirs
        out_path = os.path.join(output_path, f"training_{self.path_iter}")
        while os.path.exists(out_path):
            self.path_iter += 1
            out_path = os.path.join(output_path, f"training_{self.path_iter}")
        self.output_path = out_path

        self.model_path = os.path.join(self.output_path, "models")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.log_path = os.path.join(self.output_path, "log.csv")
        with open(self.log_path, "w") as f:
            f.write("epoch, lr, train_loss, val_loss, time\n")

        # learning rate scheduler
        self.lr_scheduler = lr_scheduler
        if lr_scheduler[0] == "plateau":
            self.scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                patience=lr_scheduler[2],
                factor=lr_scheduler[3],
                min_lr=lr_scheduler[4],
            )
            self.running_val_loss = []
        elif lr_scheduler[0] == "decay":

            def lambda1(epoch):
                return np.exp(-epoch * lr_scheduler[1])

            self.scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda1)
        else:
            raise NotImplementedError(
                f"Scheduler '{lr_scheduler[0]}' is not implemented yet."
            )

    def train(
        self,
        epochs,
        training_set,
        train_batch_size=None,
        validation_set=None,
        val_batch_size=None,
        val_freq=1,
        test_set=None,
        test_batch_size=None,
        tst_freq=10,
        clip_grad=1.0,
        max_n_neigh=None,
    ):
        # load training data
        if type(training_set) is str:
            train_data = dict(np.load(training_set, allow_pickle=True))
        elif type(training_set) is dict:
            train_data = training_set

        trn_generator, trn_steps = make_batches(
            train_data, train_batch_size, self.device[0], max_n_neigh
        )

        # load validation data
        if validation_set is not None:
            if isinstance(validation_set, str):
                val_data = dict(np.load(validation_set, allow_pickle=True))
            elif isinstance(validation_set, dict):
                val_data = validation_set
            val_generator, val_steps = make_batches(
                val_data, val_batch_size, self.device[0], max_n_neigh
            )
        else:
            val_generator = None
            val_steps = None

        # load test data
        if test_set is not None:
            if isinstance(test_set, str):
                test_data = dict(np.load(test_set, allow_pickle=True))
            elif isinstance(test_set, dict):
                test_data = test_set
            tst_generator, tst_steps = make_batches(
                test_data, test_batch_size, self.device[0], max_n_neigh
            )
        else:
            tst_generator = None
            tst_steps = None

        # move all tensors to device
        self.model.to(self.device[0])
        if (self.multi_gpu) and not self.parallelized:
            self.model = nn.DataParallel(self.model, device_ids=self.device)
            self.parallelized = True

        # move optimizer to device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device[0])

        # epoch loop
        for _ in range(epochs):
            t0 = time.time()
            self.epoch += 1

            # training
            self.model.train()
            self.optimizer.zero_grad()
            trn_loss = 0.0
            for s in range(trn_steps):
                # reset optimizer
                self.optimizer.zero_grad()

                # get new batch
                trn_batch = next(trn_generator)

                # predict
                trn_preds = self.model(trn_batch)

                # backprop
                loss = self.loss_fn(self.epoch, trn_preds, trn_batch)
                loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                self.optimizer.step()
                step_loss = loss.detach().cpu().numpy()
                trn_loss += step_loss

                print(f"\r step: {s}/{trn_steps} step_loss: {step_loss:5.5f}", end="\r")
                del trn_batch
            trn_loss /= trn_steps

            # validation
            val_loss = 0.0
            if val_generator is not None:
                if self.epoch % val_freq == 0:
                    self.model.eval()
                    for s in range(val_steps):
                        val_batch = next(val_generator)
                        val_preds = self.model(val_batch)
                        loss = self.loss_fn(self.epoch, val_preds, val_batch)
                        step_loss = loss.detach().cpu().numpy()
                        val_loss += step_loss
                        print(
                            f"\r val: {s}/{val_steps} step_loss: {step_loss:5.5f}",
                            end="\r",
                        )
                        del val_batch
                    val_loss /= val_steps
            else:
                val_loss = trn_loss

            # test
            tst_energy_mae = 0.0
            tst_forces_mae = 0.0
            if tst_generator is not None:
                if self.epoch % tst_freq == 0:
                    self.model.eval()
                    for s in range(tst_steps):
                        tst_batch = next(tst_generator)
                        tst_preds = self.model(tst_batch)

                        # energy
                        energy_err = torch.sum(
                            torch.abs(tst_preds["E"] - tst_batch["E"])
                        )
                        tst_energy_mae += energy_err.detach().cpu().numpy()

                        # forces
                        forces_err = torch.sum(
                            torch.abs(tst_preds["F"] - tst_batch["F"])
                        )
                        tst_forces_mae += forces_err.detach().cpu().numpy()

                        del tst_batch
                    tst_energy_mae /= tst_steps * test_batch_size
                    tst_forces_mae /= tst_steps * test_batch_size

            # save model with best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_model = self.model.module if self.multi_gpu else self.model
                torch.save(save_model, os.path.join(self.model_path, "best_model.pt"))
                torch.save(
                    {
                        "epoch": self.epoch,
                        "model_state_dict": save_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": loss,
                    },
                    os.path.join(self.model_path, "best_model_state.tar"),
                )

            # learning rate decay
            if self.lr_scheduler[0] == "plateau":
                self.running_val_loss.append(val_loss)
                if len(self.running_val_loss) > self.lr_scheduler[1]:
                    self.running_val_loss.pop(0)
                accum_val_loss = np.mean(self.running_val_loss)
                self.scheduler.step(accum_val_loss)
            elif self.lr_scheduler[0] == "decay":
                self.scheduler.step()
                accum_val_loss = 0.0

            # output
            """ epoch, lr, train_loss, val_loss, time"""
            for param_group in self.scheduler.optimizer.param_groups:
                lr = float(param_group["lr"])

            epoch_string = f"epoch: {self.epoch:03}/{epochs:03} "
            epoch_string += f"lr: {lr:2.2e} "
            epoch_string += f"trn_loss: {trn_loss:0.4f} "
            epoch_string += f"val_loss: {val_loss:0.4f} " if val_generator else ""
            epoch_string += f"time: {time.time() - t0:0.2f}"
            print(epoch_string)

            if (tst_generator is not None) and (self.epoch % tst_freq == 0):
                test_string = f"test_epoch: {self.epoch:3}/{epochs:3} "
                test_string += f"energy_mae: {tst_energy_mae:4.4f} "
                test_string += f"forces_mae: {tst_forces_mae:4.4f}"
                print(test_string)

            with open(self.log_path, "a") as f:
                f.write(
                    f"{self.epoch}, {lr}, {trn_loss}, {val_loss}, {time.time() - t0}\n"
                )
