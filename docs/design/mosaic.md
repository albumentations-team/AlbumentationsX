# Technical Specification: Mosaic Transform Data Handling

## 1. Objective

Design the `Mosaic` transform to correctly handle bounding box and keypoint preprocessing, including label encoding, for both the primary input image and additional data items provided via metadata. Ensure that the shared `LabelEncoder` (managed by processors provided by `Compose`) reflects the complete vocabulary from all contributing items *in the final generated mosaic instance* before final label encoding and subsequent decoding steps occur within a single `Compose` call.

## 2. Design

This workflow outlines the data processing logic within the `Mosaic` transform, primarily executed within its `sample_parameters` method or delegated helper functions/methods.

1. **Validate Raw Additional Metadata and Active Aliases:**
   * Filter the raw metadata using the canonical image/mask compatibility rules; invalid records warn and are skipped.
   * Validate every active image and semantic mask alias across the entire remaining pool before geometry or selection.
     The paired-target requirements are specified in sections 4 and 5 below.
2. **Calculate Geometry & Visible Cell Placements:**
   * Calculate the mosaic center point (`center_x`, `center_y`).
   * Determine the boundaries of the final crop window (`target_size`) relative to the conceptual large grid.
   * Calculate the placement coordinates `(x_min, y_min, x_max, y_max)` for each *visible* grid cell `(r, c)` on the final output canvas. Result: `cell_placements: dict[(r, c), tuple]`.
   * The number of keys in this dictionary defines the number of visible cells (guaranteed >= 1).

3. **Select Subset of Raw Additional Metadata:**
   * The caller supplies the complete candidate pool. Mosaic never reaches into a dataset or another global source.
   * Determine the number of *additional* items needed for the *visible* cells (`num_additional_needed = len(cell_placements) - 1`. Ensure >= 0).
   * If `len(valid_additional_raw_items) > num_additional_needed`:
     * Randomly sample `num_additional_needed` items from `valid_additional_raw_items`. Result: `selected_raw_additional_items`.
   * Else (`len <= num_additional_needed`):
     * Use every valid supplied item. Result: `selected_raw_additional_items`.

4. **Preprocess Selected Raw Additional Items (Iteratively):**
   * Access `bbox_processor` and `keypoint_processor` from `params`.
   * Initialize an empty list: `preprocessed_selected_additional_items = []`.
   * Iterate through each `item` in `selected_raw_additional_items`:
     * If `bbox_processor` exists and `item` contains 'bboxes':
       * Call `bbox_processor.preprocess(item)`. This modifies `item['bboxes']` in-place (format conversion, label encoding/updating using `item['image']` shape).
     * If `keypoint_processor` exists and `item` contains 'keypoints':
       * Call `keypoint_processor.preprocess(item)`. This modifies `item['keypoints']` in-place (format conversion, label encoding/updating using `item['image']` shape).
     * Create a `processed_item` dictionary containing the original `item['image']`, `item.get('mask')`, and the *modified* `item.get('bboxes')` and `item.get('keypoints')` from the **same `item` dictionary**. Handle cases where bboxes/keypoints might be None after processing.
     * Append `processed_item` to `preprocessed_selected_additional_items`.
   * The `preprocessed_selected_additional_items` list now contains dicts, each with the original image/mask and *preprocessed* bboxes/keypoints (in 'albumentations' format with encoded labels). The shared `LabelEncoder` within the processors has been updated with labels from all processed additional items.

5. **Prepare Primary Data:**
   * Extract the *already preprocessed* primary data from the input `data` dict (e.g., `primary = {'image': data['image'], 'mask': data.get('mask'), 'bboxes': data.get('bboxes'), ...}`).
6. **Determine Replication Count:**
   * Calculate `num_additional_needed = len(cell_placements) - 1`.
   * Calculate `num_replications = num_additional_needed - len(preprocessed_selected_additional_items)`.
7. **Replicate Primary Data:**
   * Create independent copies: `replicated_primary_items = [deepcopy(primary) for _ in range(num_replications)]`.
8. **Combine Final Preprocessed Items:**
   * Create `final_items_for_grid = [primary] + preprocessed_selected_additional_items + replicated_primary_items`.
9. **Assign Items to VISIBLE Grid Cells:**
   * Find the visible cell `(r, c)` with the largest target area from `cell_placements`.
   * Assign the `primary` (index 0 of `final_items_for_grid`) to this cell (`primary_placement_pos`).
   * Assign the remaining items (indices 1 onwards) randomly to the *remaining* `cell_placements` keys.
10. **Process Cell Geometry & Shift Coordinates:**
    * Create `processed_mosaic_pieces = {}`.
    * Iterate through the `assigned_items` (mapping `grid_pos` to the assigned `item` dict):
      * Get the final placement coordinates `placement_coords = cell_placements[grid_pos]` -> `(tgt_x1, tgt_y1, tgt_x2, tgt_y2)`.
      * Calculate target dimensions for this specific cell piece: `target_h = tgt_y2 - tgt_y1`, `target_w = tgt_x2 - tgt_x1`.
      * Get the assigned `item = final_items_for_grid[item_idx]` (containing preprocessed bboxes/kps and original image/mask).
      * Determine the cell's relative position (`cell_position`: top_left, center, etc.) on the final canvas using `get_cell_relative_position`.
      * **A. Geometric Processing (Image, Mask, BBoxes, Keypoints):**
        * Create a temporary `Compose` pipeline based on `fit_mode`:
          * If `fit_mode == "cover"`: `Compose([SmallestMaxSize(max_size_hw=cell_shape), Crop(...)], ...)`
          * If `fit_mode == "contain"`: `Compose([LongestMaxSize(max_size_hw=cell_shape), Crop(..., pad_if_needed=True)], ...)`
        * The `Crop` coordinates are determined by `get_opposite_crop_coords(cell_size=cell_shape, crop_size=(target_h, target_w), cell_position=cell_position)`.
        * Prepare the input for the pipeline: `geom_input = {'image': item['image'], ...}` including mask, bboxes, keypoints if present.
        * Apply the `geom_pipeline`: `processed_item_geom = geom_pipeline(**geom_input)`. This performs resizing (if needed) and cropping (with potential padding in "contain" mode) on image/mask, adjusting bbox/keypoint coordinates relative to the *output* of this step (which has dimensions `target_h`, `target_w`).
      * **B. Coordinate Shifting (BBoxes/Keypoints):**
        * Extract the geometrically transformed annotations (`bboxes_geom`, `keypoints_geom`) from `processed_item_geom`.
        * Denormalize `bboxes_geom` using the cell piece dimensions (`target_h`, `target_w`).
        * Apply geometric shift based on the top-left corner of `placement_coords` (`tgt_x1`, `tgt_y1`).
        * Normalize the shifted bboxes using the final `target_size` of the mosaic canvas.
        * Apply geometric shift to `keypoints_geom` (which are typically absolute pixel coords) based on `tgt_x1`, `tgt_y1`.
        * Result: `shifted_bboxes`, `shifted_keypoints`.
      * **Store Results:** Store the geometrically processed `image`/`mask` from `processed_item_geom` and the final `shifted_bboxes`/`shifted_keypoints` in a temporary dict.
      * Use the `placement_coords` tuple `(tgt_x1, tgt_y1, tgt_x2, tgt_y2)` as the key and the temporary dict as the value in `processed_mosaic_pieces`.
11. **Return Parameters:** Return `processed_mosaic_pieces` (dict mapping placement coords `(x1,y1,x2,y2)` to processed cell data `dict`).

12. **Apply Phase (`apply`, `apply_to_mask`):**
    * Receive `processed_mosaic_pieces` dict via `params`.
    * Call helper function `fmixing.assemble_mosaic_from_processed_cells`, passing `processed_mosaic_pieces`, `target_size`, number of channels of the input image/mask, `dtype`, and the `data_key` (`'image'` or `'mask'`).
    * Return the assembled canvas created by the helper function.
13. **Apply Phase (`apply_to_bboxes`, `apply_to_keypoints`):**
    * Receive `processed_mosaic_pieces` dict via `params`.
    * Initialize an empty list (e.g., `all_shifted_bboxes`).
    * Iterate through the values (cell data dicts) in `processed_mosaic_pieces`.
    * Extract the shifted `bboxes` (or `keypoints`) array from each cell data dict if it exists and is not empty.
    * Append the extracted arrays to the list.
    * If the list is empty, return an empty array with the appropriate number of columns.
    * Concatenate the list of arrays into a single NumPy array (`final_combined_bboxes` or `final_combined_keypoints`).
    * **For BBoxes:** Access the `bbox_processor`. Call `bbox_utils.filter_bboxes` on `final_combined_bboxes` using `target_size` and the filtering parameters (`min_area`, `min_visibility`, etc.) obtained from `bbox_processor.params`.
    * **For Keypoints:** Perform boundary checking against `target_size` to filter out keypoints outside the final canvas.
    * Return the final concatenated and filtered array.

## 3. Notes on Label Encoding

* **`Compose.preprocess`:** When the main `Compose` pipeline runs, it preprocesses the *primary* input data. If labels are non-numeric, `bbox_processor` and `keypoint_processor` fit an internal `LabelEncoder` based *only* on the labels found in this primary data.
* **`Mosaic` Preprocessing (Step 4):** `Mosaic` then preprocesses the *selected raw additional items*. The *same* processor instances are used. If these items contain labels not seen in the primary data, the processor's internal `LabelEncoder.update` method is called (implicitly via `processor.preprocess`). This extends the encoder's vocabulary without changing existing encodings. The encoder's state now reflects labels from the primary item *plus* the selected additional items for this specific mosaic instance.
* **`Compose.postprocess`:** After all transforms (including `Mosaic`) complete, `Compose.postprocess` uses the *final state* of the processors (including the updated `LabelEncoder`) to decode all labels present in the output data back to their original format. Because the encoder was updated during the `Mosaic` step with all labels *actually included* in that mosaic instance, the decoding should be correct for that instance's output.
* **Scope:** The `LabelEncoder` state is transient *per `Compose` call*. It does not persist across different calls or build a vocabulary over the entire dataset. Its purpose is to handle non-numeric labels correctly *within* a single augmentation pipeline run.

## 4. Paired additional image targets

An active `additional_targets={"thermal": "image"}` target must use the pixels from `thermal`
in the primary sample and the same selected donor records as `image`. Each target retains its own
dtype and channels. The public HW/HWC1 representation is restored by the usual invocation boundary.

Before sampling geometry or selecting surplus donors, Mosaic validates every canonical-valid donor
for each active image alias. The donor alias must be a non-empty HW/HWC NumPy array, have the same
H/W as that donor's image, and match the primary alias's dtype and channel count. HW and HWC1 are
compatible. Supported alias dtypes are uint8 and float32; the canonical image and aliases may have
different dtypes. Tuple fill must match every active image target's channels; scalar fill is shared.
Active image aliases with more than 128 channels are rejected before geometry and donor selection,
including when no resize would be needed. This bound avoids the OpenCV failure in the cell resize
path; both the primary alias and each canonical-valid donor are checked.
Missing or incompatible alias data raises ValueError rather than substituting RGB or sampling another
donor pool. Canonical-invalid metadata retains the existing warn-and-skip behavior. An alias registered
but absent from the current primary call is inactive and places no requirements on donor records.

The internal `ProcessedMosaicItem.additional_images` mapping carries alias arrays through primary
replication, donor annotation preprocessing, cell geometry and coordinate shifting. Each cell's
existing Compose receives these arrays as additional image targets, so resizing/cropping/padding
are shared and annotation preprocessing is performed once. Shared parameters hold the processed cells;
`TargetParams` supplies each image target's own canvas shape and alias key to the assembly operation.
The canonical route keeps its existing shared parameters when no image aliases are active.

Replay of the recorded parameters retains target shape/dtype requirements. This does not broaden the
applied-configuration replay guarantee or introduce Tensor donor metadata, raw uint16 image aliases
support or a new multimodal metadata schema.

## 5. Paired semantic mask aliases

An active `additional_targets={"auxmask": "mask"}` target uses its own primary and donor arrays.
An alias is active only when supplied and non-None in the current call; registration alone imposes
no donor requirement. Compose still requires the canonical `mask` when a mask alias is supplied.
The supported direct route is `Mosaic.add_targets(...)` followed by a normal transform call with
the canonical mask and its aliases. Direct calls reject an omitted or None primary canonical mask
with ValueError before geometry or donor selection, for both scalar and tuple `fill_mask`.

After canonical metadata filtering, and before any Mosaic geometry or donor sampling, validate
every canonical-valid donor, including surplus candidates and donors unused by a 1x1 mosaic:

- Every donor must have a non-None canonical `mask` whenever a semantic mask alias is active.
  Omitted or None canonical donor masks raise ValueError identifying the donor and active aliases.
  No placeholder mask or alias substitution is used. When no semantic mask alias is active,
  omitted/None donor masks retain the existing `fill_mask` behavior, including image-alias-only calls.
- Every active alias must be a non-empty NumPy HW/HWC array, spatially aligned with its own image,
  and have the same dtype and channel count as that alias in the primary sample. Primary aliases
  are validated too. Different targets may have different dtypes and channels; HW/HWC1 are equivalent.
  Missing or incompatible aliases raise ValueError identifying the target and donor position in
  the canonical-valid pool. There is no alias-specific filtering, replacement, or silent cast.
- Semantic aliases support 1-128 channels. Wider aliases fail before sampling even for identity
  geometry. This bounds the existing OpenCV cell resize/pad route; canonical-mask-only behavior
  is unchanged.
- Tuple `fill_mask` must match each supplied semantic target, including canonical `mask`.
  Use scalar fill for mixed channel counts. Instance-stack fill policy is unchanged.

Masks use `mask_interpolation` and `fill_mask` through the canonical mask geometry path.
They inherit its dtype/interpolation support, without the image-alias uint8/float32 restriction.
For example, uint16/int16 masks retain labels above 255 or below zero; int32 nearest resize works,
while int32 linear resize retains the canonical OpenCV rejection. This adds no dtype conversion
or support for combinations that the canonical mask operation rejects.

`ProcessedMosaicItem.additional_masks` carries references through primary/donor preparation
and existing primary replication, then each cell's existing Compose handles the aliases as masks.
Geometry, donor selection, and annotation preprocessing are shared. Coordinate shifting,
instance-ID remapping, and bbox filtering preserve the semantic payload without filtering or
renumbering its pixels as instance-mask rows. Each mask's `TargetParams` provides its own canvas
shape and alias key; assembly allocates an independent output with the primary target's dtype
and public HW/HWC1/HWC representation.

Caller arrays and donor dictionaries, including read-only arrays and views, remain unchanged.
Applied outputs share no mutable storage with inputs or previous outputs. Skipped Mosaic calls
retain the existing identity behavior and do not validate donor pairs eagerly. With p=1 throughout,
pair errors occur before geometry/selection RNG use; no rollback of outer probability draws or
earlier transforms is promised.

Image and mask aliases can be combined. Constructor serialization and ReplayCompose captured
parameters retain their existing contracts; mask target requirements record shape and dtype.
The existing root Tensor fallback accepts primary mask aliases with its supported Tensor dtypes
and restores each target's container/rank; donor aliases remain NumPy arrays. No new Tensor donor
schema, instance `masks` alias support, or stronger applied-configuration replay is introduced.
