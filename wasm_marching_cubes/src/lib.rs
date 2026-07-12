use std::slice;
use mcubes::{MarchingCubes, MeshSide};
use lin_alg::f32::Vec3;

// 256^3 bytes = 16.7 MB
static mut VOLUME_BUFFER: [u8; 16_777_216] = [0; 16_777_216];
static mut VERTEX_BUFFER: [f32; 3_000_000] = [0.0; 3_000_000];
static mut INDEX_BUFFER: [u32; 3_000_000] = [0; 3_000_000];

#[no_mangle]
pub extern "C" fn get_volume_ptr() -> *mut u8 {
    unsafe { VOLUME_BUFFER.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn get_vertex_ptr() -> *mut f32 {
    unsafe { VERTEX_BUFFER.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn get_index_ptr() -> *mut u32 {
    unsafe { INDEX_BUFFER.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn run_marching_cubes(
    width: i32,
    height: i32,
    depth: i32,
    level: f32,
    step_size: i32,
) -> u64 {
    let w = (width / step_size) as usize;
    let h = (height / step_size) as usize;
    let d = (depth / step_size) as usize;
    let step = step_size as usize;

    let mut values = Vec::with_capacity(w * h * d);

    // Read downsampled volume into flat values list
    unsafe {
        let vol_ptr = VOLUME_BUFFER.as_ptr();
        for x in 0..w {
            for y in 0..h {
                for z in 0..d {
                    let orig_x = x * step;
                    let orig_y = y * step;
                    let orig_z = z * step;

                    let idx = orig_z + orig_y * (depth as usize) + orig_x * (height as usize) * (depth as usize);
                    let val = *vol_ptr.add(idx) as f32;
                    values.push(val);
                }
            }
        }
    }

    // Initialize MarchingCubes
    // nx, ny, nz are grid points
    // cell sizes are 1.0
    // min coordinates are 0.0
    // step is Vec3::new(1.0, 1.0, 1.0)
    let mc_res = MarchingCubes::new(
        (w, h, d),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0),
        values,
        level,
    );

    let mc = match mc_res {
        Ok(m) => m,
        Err(_) => return 0,
    };

    let mesh = mc.generate(MeshSide::Both);

    let v_len = mesh.vertices.len();
    let i_len = mesh.indices.len();

    if v_len * 3 > 3_000_000 || i_len > 3_000_000 {
        return 0;
    }

    unsafe {
        let step_f = step as f32;
        for i in 0..v_len {
            let pos = mesh.vertices[i].posit;
            VERTEX_BUFFER[i * 3]     = pos.x as f32 * step_f;
            VERTEX_BUFFER[i * 3 + 1] = pos.y as f32 * step_f;
            VERTEX_BUFFER[i * 3 + 2] = pos.z as f32 * step_f;
        }

        // Copy indices with usize to u32 cast
        for i in 0..i_len {
            INDEX_BUFFER[i] = mesh.indices[i] as u32;
        }
    }

    ((v_len as u64) << 32) | (i_len as u64)
}
