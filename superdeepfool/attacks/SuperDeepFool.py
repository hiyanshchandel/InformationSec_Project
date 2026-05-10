import numpy as np
import torch
import torch.nn as nn

from superdeepfool.attacks.attack import Attack
from superdeepfool.attacks.DeepFool import DeepFool

try:
    from curvature.curvature import curvature_hessian_estimator
except Exception:
    curvature_hessian_estimator = None


class SuperDeepFool(Attack):
    def __init__(
        self,
        model,
        steps: int = 100,
        overshoot: float = 0.02,
        search_iter: int = 0,
        number_of_samples=None,
        l_norm: str = "L2",
        curvature_lambda: float = 0.0,
        curvature_power_iter: int = 1,
    ):
        super().__init__("SuperDeepFool", model)
        self.steps = steps
        self.overshoot = overshoot
        self.deepfool = DeepFool(
            model, steps=steps, overshoot=overshoot, search_iter=10
        )
        self._supported_mode = ["default"]
        self.search_iter = search_iter
        self.number_of_samples = number_of_samples
        self.fool_checker = 0
        self.l_norm = l_norm
        self.target_label = None
        self.curvature_lambda = curvature_lambda
        self.curvature_power_iter = curvature_power_iter

    def forward(self, images, labels, verbose: bool = True):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = len(images)
        correct = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        curr_steps = 0
        r_tot = torch.zeros_like(images)
        adv_images = images.clone().detach()

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                if not correct[idx]:
                    continue

                image = images[idx : idx + 1]
                label = labels[idx : idx + 1]
                label_idx = int(label.item())
                r_ = r_tot[idx : idx + 1]
                adv_image = adv_images[idx : idx + 1]

                fs = self.model(adv_image)[0]
                _, pre = torch.max(fs, dim=0)
                if int(pre.item()) != label_idx:
                    correct[idx] = False
                    continue

                adv_image_Deepfool, target_label = self.deepfool(
                    adv_image, label, return_target_labels=True
                )
                r_i = adv_image_Deepfool - image
                adv_image_Deepfool.requires_grad = True
                fs = self.model(adv_image_Deepfool)[0]
                _, pre = torch.max(fs, dim=0)

                target_label = target_label.detach()
                target_label_idx = int(target_label.item())
                pre_idx = int(pre.item())
                if pre_idx == label_idx:
                    pre_idx = target_label_idx
                cost = fs[pre_idx] - fs[label_idx]

                last_grad = torch.autograd.grad(
                    cost, adv_image_Deepfool, retain_graph=False, create_graph=False
                )[0]

                if self.l_norm == "L2":
                    last_grad_norm = last_grad.norm(p=2).clamp_min(1e-12)
                    update = (last_grad * r_i).sum() * last_grad / (last_grad_norm**2)

                    if self.curvature_lambda > 0 and curvature_hessian_estimator is not None:
                        curvature, _, _ = curvature_hessian_estimator(
                            self.model,
                            adv_image_Deepfool.detach(),
                            label,
                            num_power_iter=self.curvature_power_iter,
                        )
                        curvature_scale = 1.0 / (1.0 + self.curvature_lambda * curvature.clamp_min(0.0))
                        update = update * curvature_scale.view(-1, 1, 1, 1)

                    r_ = r_ + update

                adv_image = image + r_
                adv_images[idx : idx + 1] = adv_image.detach()
                r_tot[idx : idx + 1] = r_.detach()
                self.target_label = target_label.detach()

            curr_steps += 1

        adv_images = adv_images.detach()
        if self.search_iter > 0:
            if verbose:
                print(f"search iteration for SuperDeepfool -> {self.search_iter}")
            dx = adv_images - images
            dx_l_low, dx_l_high = torch.zeros_like(dx), torch.ones_like(dx)
            for i in range(self.search_iter):
                dx_l = (dx_l_low + dx_l_high) / 2.0
                dx_x = images + dx_l * dx
                dx_y = self.model(dx_x).argmax(-1)
                label_stay = dx_y == labels
                label_change = dx_y != labels
                dx_l_low[label_stay] = dx_l[label_stay]
                dx_l_high[label_change] = dx_l[label_change]
            adv_images = images + dx_l_high * dx
        return adv_images
