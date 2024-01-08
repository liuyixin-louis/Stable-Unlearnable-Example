# Stable-Unlearnable-Example
[Paper](https://arxiv.org/abs/2311.13091)
This is official implementation of AAAI'24 Stable Unlearnable Example: Enhancing the Robustness of Unlearnable Examples via Stable Error-Minimizing Noise. 

![SEM-framework](./SEM-framework.jpg)

The complete code is coming soon. On the implementation level, our work achieves dual improvement in effectiveness and efficiency by solely overwriting the `_get_adv_` [function](https://github.com/fshp971/robust-unlearnable-examples/blob/main/attacks/robust_workers.py) in `robust-unlearnable-examples/attacks/robust_workers.py` in the [REM code base](https://github.com/fshp971/robust-unlearnable-examples) with the following random perturbation process. You can simply setup the codebase following REM and apply the following modification. 

```python
def _get_adv_(self, model, criterion, x, y,):
  adv_x = x.clone()
  if self.atk_steps==0 or self.atk_radius==0:
      return adv_x
  # "uniform" noise
  adv_x += 2 * (torch.rand_like(x) - 0.5) * self.atk_radius * self.uniform_scale
  self._clip_(adv_x, x, radius=self.atk_radius)
  return adv_x.data
```
