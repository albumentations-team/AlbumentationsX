from __future__ import annotations
import inspect

def candidates(f):
    original=f.generate_plasma_pattern
    text=inspect.getsource(original)
    def make(source):
        ns=vars(f).copy()
        exec(compile(source,'<candidate>','exec',flags=0x1000000),ns)
        return ns[original.__name__]
    in_place=text
    for step in ('diamond','square'):
        line=f'        expanded_grid += ({step}_interpolation + all_noise) * {step}_mask'
        subst=f'''        np.add({step}_interpolation, all_noise, out={step}_interpolation)
        np.multiply({step}_interpolation, {step}_mask, out={step}_interpolation)
        expanded_grid += {step}_interpolation'''
        assert line in in_place
        in_place=in_place.replace(line,subst)
    in_place=in_place.replace('return _normalize_minmax_float32(expanded_grid)','cv2.normalize(expanded_grid, expanded_grid, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)\n        return expanded_grid')
    a=in_place.index('    return np.clip(')
    in_place=in_place[:a]+'''    pattern = _normalize_minmax_float32(plasma_grid[: target_shape[0], : target_shape[1]])
    return np.clip(pattern, 0, 1, out=pattern)
'''
    reusable=in_place.replace('        # Diamond step', '''        work = filter_buffer[:expanded_grid.size].reshape(expanded_grid.shape)
        mask = mask_buffer[:expanded_grid.size].reshape(expanded_grid.shape)
        # Diamond step''')
    for step in ('diamond','square'):
        reusable=reusable.replace(f'{step.upper()}_KERNEL, borderType',f'{step.upper()}_KERNEL, dst=work, borderType')
        reusable=reusable.replace(f'{step}_mask = {step}_interpolation > 0',f'np.greater({step}_interpolation, 0, out=mask)')
        reusable=reusable.replace(f', {step}_mask, out',', mask, out')
    reusable=reusable.replace('    # Recursively apply diamond-square steps', '''    workspace_size = int(power_of_two_size) ** 2
    filter_buffer = np.empty(workspace_size, dtype=np.float32)
    mask_buffer = np.empty(workspace_size, dtype=bool)

    # Recursively apply diamond-square steps''')
    uniform=reusable.replace('all_noise = random_generator.uniform(-noise_scale, noise_scale, (next_height, next_width)).astype(np.float32)', '''noise = random_buffer[:next_height * next_width].reshape(next_height, next_width)
        random_generator.random(out=noise)
        noise *= 2 * float(noise_scale)
        noise -= float(noise_scale)
        all_noise = noise.astype(np.float32)''').replace('    filter_buffer = np.empty','    random_buffer = np.empty(workspace_size, dtype=np.float64)\n    filter_buffer = np.empty')
    compact=reusable.replace('    mask_buffer = np.empty(workspace_size, dtype=bool)','    mask_buffer = np.empty(workspace_size, dtype=np.uint8)').replace('np.greater(diamond_interpolation, 0, out=mask)','cv2.compare(diamond_interpolation, 0, cv2.CMP_GT, dst=mask)').replace('np.greater(square_interpolation, 0, out=mask)','cv2.compare(square_interpolation, 0, cv2.CMP_GT, dst=mask)')
    for step in ('diamond','square'):
        compact=compact.replace(f'        np.multiply({step}_interpolation, mask, out={step}_interpolation)\n        expanded_grid += {step}_interpolation', f'        np.add(expanded_grid, {step}_interpolation, out=expanded_grid, where=mask.view(bool))')
    return {n:make(s) for n,s in [('base',text),('inplace',in_place),('reuse',reusable),('random',uniform),('compare',compact)]}
