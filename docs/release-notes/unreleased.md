# Unreleased

- Fixed delayed release of input arrays and CPU tensors after `Compose` calls,
  including calls that raise an exception. Optional-channel images, masks,
  batches and volumes no longer depend on cyclic garbage collection for release.
  This addresses [#596](https://github.com/albumentations-team/AlbumentationsX/issues/596).
- **Custom sampler compatibility:** `SamplingContext.invocation` and the
  `invocation=` constructor argument have been removed. Use `sampling.py_random`,
  `sampling.random_generator`, the sampling hook's `data` and `targets`, or
  `SamplingContext.from_seed()` / `from_owner()`. See the
  [migration guide](../contributing/coding_guidelines.md#samplingcontext-compatibility-and-migration).
