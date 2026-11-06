# Pipeline validation and errors

If a mask has a different size from its image, `Compose` raises before deciding whether to augment the sample. Fixing
the sample and retrying with the same pipeline uses the same random sequence as calling it with valid data in the
first place. This also applies when the pipeline's probability is zero and when you call `run_with_trace()`.

## Which exception to expect

| Exception | What to correct |
| --- | --- |
| `TypeError` | Pass the expected Python type, array or Tensor container, or rank. For example, pass an image array with height and width axes, with an optional channel axis. |
| `ValueError` | Correct an invalid value or configuration, supply a required field, or align incompatible shapes. For example, map an alias to `"image"` and supply both the alias and `image`. |
| `RuntimeError` | Report a failed execution invariant if it occurs through a supported call. Include the pipeline, input shapes and dtypes, and traceback. |
| Compatibility warning | Follow the replacement in the warning. A deprecation notice gives the condition for removing the old behavior. |

Input errors identify the argument, target, or field, the expected contract, and the received value, kind, or shape
where safe. They give you a correction without requiring you to inspect the implementation.

## Where validation happens

| Boundary | What it checks |
| --- | --- |
| Transform construction | Declared parameters and relationships between them. Invalid declared parameters raise in both strict modes. |
| `Compose` construction | Graph configuration, probabilities, additional-target mappings, and processor compatibility such as OBB support. |
| Root `Compose` call | Target containers and ranks, CPU Tensor constraints, required sources for spatial aliases, and compatible spatial sizes. Configured processors also check annotation rows, coordinates and label alignment before the probability draw. Empty annotations are accepted. |
| Replay and deserialization | Required fields, field kinds, and transform registration. A replay checks the new sample against the saved parameter requirements when it reaches the corresponding transform. |
| Transform execution | Conditions that depend on an intermediate target, a selected branch, or realized parameters. A failure here can occur after random draws have already been used. |

Shape checks compare metadata. Coordinate processors reject non-finite coordinates. Image and volume pixel values
follow the input contract of the selected operation.

## Strict mode and compatibility

- `A.Blur(strict=False, blurr_range=(3, 5))` warns about the unknown keyword. Rename it to `blur_range`.
  Use `strict=True` to make unknown transform keywords errors.
- `Compose(strict=True)` checks transform keywords and rejects unknown sample keys. With `strict=False`,
  unrelated sample keys can pass through and `None` on an optional mask target means the mask is absent.
- `is_check_shapes=False` permits deliberately different spatial sizes. Container and rank checks still apply.
- A skipped valid call returns its original targets. Configuring a probability of zero still runs the input checks.

If you handle errors from an older version, catch `TypeError` for positional augmentation inputs and
`ValueError` for missing replay fields or unknown serialized transform names. Pass inputs as named arguments
and use a complete payload from a compatible pipeline.
