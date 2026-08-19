use std::env;
use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use glfw::{Action, Context, Key, WindowHint};
use rayon::prelude::*;

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const FRAME_SIZE: usize = WIDTH * HEIGHT * 3; // RGB24 frame size = 2,764,800 bytes
const MAGIC: &[u8; 4] = b"GLZ4";

fn get_lz4_dataset_dir() -> PathBuf {
    if let Ok(env_val) = std::env::var("GAIM240_LZ4_DIR") {
        let p = PathBuf::from(env_val);
        if p.exists() {
            return p;
        }
    }
    for fallback in &[
        "/home/jv495/Downloads/GAIM240_refs_lz4",
        "../GAIM240_refs_lz4",
        "../../Downloads/GAIM240_refs_lz4",
        "D:\\GAIM240_refs_lz4",
        "C:\\GAIM240_refs_lz4",
    ] {
        let p = PathBuf::from(fallback);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("GAIM240_refs_lz4")
}

struct Lz4StreamData {
    num_frames: usize,
    raw_data: Vec<u8>,
}

fn load_and_decompress_lz4_file(path: &Path) -> Result<Lz4StreamData, String> {
    let t_io = Instant::now();
    let file_bytes = fs::read(path).map_err(|e| format!("Failed to read {:?}: {}", path, e))?;
    let io_dur = t_io.elapsed().as_secs_f64();

    if file_bytes.len() < 24 {
        return Err(format!("File too small for header: {:?}", path));
    }

    if &file_bytes[0..4] != MAGIC {
        return Err(format!("Invalid magic bytes in {:?}", path));
    }

    let version = u32::from_le_bytes(file_bytes[4..8].try_into().unwrap());
    if version != 1 {
        return Err(format!("Unsupported version {} in {:?}", version, path));
    }

    let width = u32::from_le_bytes(file_bytes[8..12].try_into().unwrap()) as usize;
    let height = u32::from_le_bytes(file_bytes[12..16].try_into().unwrap()) as usize;
    let num_frames = u32::from_le_bytes(file_bytes[16..20].try_into().unwrap()) as usize;
    let frame_size = u32::from_le_bytes(file_bytes[20..24].try_into().unwrap()) as usize;

    if width != WIDTH || height != HEIGHT || frame_size != FRAME_SIZE {
        return Err(format!("Dimensions mismatch in {:?}: {}x{} frame_size={}", path, width, height, frame_size));
    }

    let table_offset = 24;
    let payload_offset = table_offset + num_frames * 4;
    if file_bytes.len() < payload_offset {
        return Err(format!("Corrupted header in {:?}", path));
    }

    let mut compressed_sizes = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = table_offset + i * 4;
        let size = u32::from_le_bytes(file_bytes[start..start + 4].try_into().unwrap()) as usize;
        compressed_sizes.push(size);
    }

    let mut frame_slices = Vec::with_capacity(num_frames);
    let mut curr_offset = payload_offset;
    for &size in &compressed_sizes {
        if curr_offset + size > file_bytes.len() {
            return Err(format!("Truncated payload in {:?}", path));
        }
        frame_slices.push(&file_bytes[curr_offset..curr_offset + size]);
        curr_offset += size;
    }

    let t_decomp = Instant::now();
    let mut raw_data = vec![0u8; num_frames * FRAME_SIZE];

    // Parallel multi-core zero-allocation decompression
    raw_data
        .par_chunks_exact_mut(FRAME_SIZE)
        .zip(frame_slices.par_iter())
        .for_each(|(target, slice)| {
            lz4_flex::decompress_into(&slice[4..], target).expect("LZ4 frame decompression failed");
        });

    let decomp_dur = t_decomp.elapsed().as_secs_f64();
    println!("  [{:?}] Read: {:.3}s ({:.1} MB/s) | Decompress: {:.3}s ({:.1} GB/s)", 
        path.file_name().unwrap(), io_dur, (file_bytes.len() as f64 / 1024.0 / 1024.0) / io_dur, decomp_dur, (raw_data.len() as f64 / 1024.0 / 1024.0 / 1024.0) / decomp_dur);

    Ok(Lz4StreamData { num_frames, raw_data })
}

fn create_empty_rgb_texture() -> u32 {
    let mut tex = 0;
    unsafe {
        gl::GenTextures(1, &mut tex);
        gl::BindTexture(gl::TEXTURE_2D, tex);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexImage2D(
            gl::TEXTURE_2D, 0, gl::RGB as i32, WIDTH as i32, HEIGHT as i32, 0,
            gl::RGB, gl::UNSIGNED_BYTE, std::ptr::null()
        );
    }
    tex
}

unsafe fn upload_rgb_frame(tex: u32, raw_data: &[u8]) {
    gl::ActiveTexture(gl::TEXTURE0);
    gl::BindTexture(gl::TEXTURE_2D, tex);
    gl::TexSubImage2D(
        gl::TEXTURE_2D, 0, 0, 0, WIDTH as i32, HEIGHT as i32,
        gl::RGB, gl::UNSIGNED_BYTE, raw_data.as_ptr() as *const _
    );
}

struct RgbQuadShader {
    program: u32,
    vbo: u32,
    u_tex_loc: i32,
}

impl RgbQuadShader {
    fn new() -> Self {
        let vert_code = CString::new(
            "
            #version 120
            attribute vec2 position;
            attribute vec2 texcoord;
            varying vec2 v_texcoord;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                v_texcoord = texcoord;
            }
        ",
        )
        .unwrap();

        let frag_code = CString::new(
            "
            #version 120
            uniform sampler2D u_tex;
            varying vec2 v_texcoord;
            void main() {
                gl_FragColor = texture2D(u_tex, v_texcoord);
            }
        ",
        )
        .unwrap();

        unsafe {
            let vert_shader = gl::CreateShader(gl::VERTEX_SHADER);
            gl::ShaderSource(vert_shader, 1, &vert_code.as_ptr(), std::ptr::null());
            gl::CompileShader(vert_shader);

            let frag_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
            gl::ShaderSource(frag_shader, 1, &frag_code.as_ptr(), std::ptr::null());
            gl::CompileShader(frag_shader);

            let program = gl::CreateProgram();
            gl::AttachShader(program, vert_shader);
            gl::AttachShader(program, frag_shader);
            gl::LinkProgram(program);

            let mut vbo = 0;
            gl::GenBuffers(1, &mut vbo);

            let u_tex_loc = gl::GetUniformLocation(program, CString::new("u_tex").unwrap().as_ptr());

            Self {
                program,
                vbo,
                u_tex_loc,
            }
        }
    }

    fn draw_quad(&self, x1: f32, y1: f32, x2: f32, y2: f32, tex: u32) {
        #[repr(C)]
        struct Vertex {
            pos: [f32; 2],
            uv: [f32; 2],
        }

        let vertices: [Vertex; 6] = [
            Vertex { pos: [x1, y1], uv: [0.0, 1.0] },
            Vertex { pos: [x2, y1], uv: [1.0, 1.0] },
            Vertex { pos: [x2, y2], uv: [1.0, 0.0] },
            Vertex { pos: [x1, y1], uv: [0.0, 1.0] },
            Vertex { pos: [x2, y2], uv: [1.0, 0.0] },
            Vertex { pos: [x1, y2], uv: [0.0, 0.0] },
        ];

        unsafe {
            gl::UseProgram(self.program);
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, tex);
            gl::Uniform1i(self.u_tex_loc, 0);

            gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (vertices.len() * std::mem::size_of::<Vertex>()) as isize,
                vertices.as_ptr() as *const _,
                gl::DYNAMIC_DRAW,
            );

            let pos_attr = CString::new("position").unwrap();
            let pos_loc = gl::GetAttribLocation(self.program, pos_attr.as_ptr());
            gl::EnableVertexAttribArray(pos_loc as u32);
            gl::VertexAttribPointer(
                pos_loc as u32,
                2,
                gl::FLOAT,
                gl::FALSE,
                std::mem::size_of::<Vertex>() as i32,
                std::ptr::null(),
            );

            let uv_attr = CString::new("texcoord").unwrap();
            let uv_loc = gl::GetAttribLocation(self.program, uv_attr.as_ptr());
            gl::EnableVertexAttribArray(uv_loc as u32);
            gl::VertexAttribPointer(
                uv_loc as u32,
                2,
                gl::FLOAT,
                gl::FALSE,
                std::mem::size_of::<Vertex>() as i32,
                (2 * std::mem::size_of::<f32>()) as *const _,
            );

            gl::DrawArrays(gl::TRIANGLES, 0, 6);

            gl::DisableVertexAttribArray(pos_loc as u32);
            gl::DisableVertexAttribArray(uv_loc as u32);
        }
    }
}

fn render_pyramid(
    window: &mut glfw::Window,
    shader: &RgbQuadShader,
    tex_left: u32,
    tex_ref: u32,
    tex_right: u32,
) {
    let (w, h) = window.get_size();

    let target_aspect = 16.0 / 9.0;
    let (w_view, h_view) = if (w as f32 / h as f32) > target_aspect {
        let h_view = h;
        let w_view = (h as f32 * target_aspect) as i32;
        (w_view, h_view)
    } else {
        let w_view = w;
        let h_view = (w as f32 / target_aspect) as i32;
        (w_view, h_view)
    };

    let x_offset = (w - w_view) / 2;
    let y_offset = (h - h_view) / 2;

    unsafe {
        gl::Viewport(0, 0, w, h);
        gl::ClearColor(0.02, 0.02, 0.03, 1.0);
        gl::Clear(gl::COLOR_BUFFER_BIT);

        gl::Viewport(x_offset, y_offset, w_view, h_view);

        shader.draw_quad(-0.5, 0.0, 0.5, 1.0, tex_ref);
        shader.draw_quad(-1.0, -1.0, 0.0, 0.0, tex_left);
        shader.draw_quad(0.0, -1.0, 1.0, 0.0, tex_right);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut no_vsync = false;
    let mut use_pacer = false;

    for arg in &args[1..] {
        if arg == "--no-vsync" || arg == "--novsync" || arg == "--uncapped" {
            no_vsync = true;
        } else if arg == "--pacer" || arg == "--pace-240" {
            no_vsync = true;
            use_pacer = true;
        }
    }

    let dataset_dir = get_lz4_dataset_dir();
    if !dataset_dir.exists() {
        println!("Error: Lossless LZ4 dataset directory not found at {:?}", dataset_dir);
        return;
    }

    let mut lz4_files = Vec::new();
    if let Ok(entries) = fs::read_dir(&dataset_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() && p.extension().map_or(false, |ext| ext == "lz4") {
                lz4_files.push(p);
            }
        }
    }
    lz4_files.sort();

    if lz4_files.len() < 3 {
        println!("Error: Need at least 3 .lz4 files inside {:?}, found {}", dataset_dir, lz4_files.len());
        return;
    }

    let left_path = lz4_files[0].clone();
    let ref_path = lz4_files[1].clone();
    let right_path = lz4_files[2].clone();

    println!("\n=================================================================");
    println!("  GAIM240 LOSSLESS LZ4 SEQUENCE BENCHMARK PLAYER");
    println!("=================================================================");
    println!("Dataset Dir     : {:?}", dataset_dir);
    println!("Left Stream Path: {:?}", left_path.file_name().unwrap());
    println!("Ref Stream Path : {:?}", ref_path.file_name().unwrap());
    println!("Right Stream Path: {:?}", right_path.file_name().unwrap());
    println!(
        "VSync / Pacer   : {}",
        if use_pacer {
            "SOFTWARE PACER (--pacer 240.000 FPS Locked)"
        } else if no_vsync {
            "UNCAPPED (--no-vsync Raw FPS Max Throughput)"
        } else {
            "HARDWARE VSYNC (Sync 1 240Hz Locked)"
        }
    );
    println!("=================================================================\n");

    // 1. Parallel loading & decompression of all 3 streams concurrently
    println!("Step 1: Reading .lz4 streams & decompressing to RAM in parallel...");
    let t_load_start = Instant::now();

    let (left_res, (ref_res, right_res)) = rayon::join(
        || load_and_decompress_lz4_file(&left_path),
        || rayon::join(
            || load_and_decompress_lz4_file(&ref_path),
            || load_and_decompress_lz4_file(&right_path)
        )
    );

    let left_stream = left_res.expect("Failed to decode left LZ4 stream");
    let ref_stream = ref_res.expect("Failed to decode ref LZ4 stream");
    let right_stream = right_res.expect("Failed to decode right LZ4 stream");

    let t_load = t_load_start.elapsed().as_secs_f64();
    let num_frames = left_stream.num_frames.min(ref_stream.num_frames).min(right_stream.num_frames);
    let total_ram_mb = (num_frames * 3 * FRAME_SIZE) as f64 / 1024.0 / 1024.0;
    let decompress_throughput_gb_s = (total_ram_mb / 1024.0) / t_load;

    println!(
        "Pre-decode Complete: Loaded {} frames in {:.4}s ({:.2} MB in RAM, {:.2} GB/s decompress throughput)",
        num_frames * 3,
        t_load,
        total_ram_mb,
        decompress_throughput_gb_s
    );

    // 2. Initialize GLFW and OpenGL
    println!("Step 2: Initializing graphics context...");
    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    glfw.window_hint(WindowHint::Resizable(true));
    glfw.window_hint(WindowHint::DoubleBuffer(true));
    glfw.window_hint(WindowHint::RefreshRate(Some(240)));
    glfw.window_hint(WindowHint::ContextVersion(2, 1));

    let (mon_w, mon_h) = glfw.with_connected_monitors(|_, monitors| {
        if let Some(mon) = monitors.first() {
            if let Some(mode) = mon.get_video_mode() {
                return (mode.width, mode.height);
            }
        }
        (1280, 720)
    });

    let (mut window, events) = glfw.with_connected_monitors(|glfw_ref, monitors| {
        if let Some(mon) = monitors.first() {
            glfw_ref
                .create_window(mon_w, mon_h, "GAIM240 Lossless LZ4 Visualizer", glfw::WindowMode::FullScreen(mon))
                .unwrap()
        } else {
            glfw_ref
                .create_window(1280, 720, "GAIM240 Lossless LZ4 Visualizer", glfw::WindowMode::Windowed)
                .unwrap()
        }
    });

    window.make_current();
    window.set_key_polling(true);
    if no_vsync {
        glfw.set_swap_interval(glfw::SwapInterval::None);
    } else {
        glfw.set_swap_interval(glfw::SwapInterval::Sync(1));
    }

    gl::load_with(|s| window.get_proc_address(s) as *const _);

    let shader = RgbQuadShader::new();
    let tex_left = create_empty_rgb_texture();
    let tex_ref = create_empty_rgb_texture();
    let tex_right = create_empty_rgb_texture();

    // GPU and Shader Warmup Phase
    {
        println!("Warming up GPU and compiling shaders...");
        let dummy_data = vec![0u8; FRAME_SIZE];
        for _ in 0..60 {
            unsafe {
                upload_rgb_frame(tex_left, &dummy_data);
                upload_rgb_frame(tex_ref, &dummy_data);
                upload_rgb_frame(tex_right, &dummy_data);
                render_pyramid(&mut window, &shader, tex_left, tex_ref, tex_right);
            }
            window.swap_buffers();
            glfw.poll_events();
            std::thread::sleep(std::time::Duration::from_millis(4));
        }
        println!("GPU Warmup complete.");
    }

    // 3. Playback Presentation Loop
    println!("Step 3: Starting playback loop...");
    let start_time = Instant::now();
    let mut swap_timestamps: Vec<Instant> = Vec::new();
    let mut step = 0usize;
    let mut quit = false;

    while !window.should_close() && step < num_frames && !quit {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            if let glfw::WindowEvent::Key(Key::Escape | Key::Q, _, Action::Press, _) = event {
                quit = true;
            }
        }

        let offset = step * FRAME_SIZE;
        let frame_left = &left_stream.raw_data[offset..offset + FRAME_SIZE];
        let frame_ref = &ref_stream.raw_data[offset..offset + FRAME_SIZE];
        let frame_right = &right_stream.raw_data[offset..offset + FRAME_SIZE];

        unsafe {
            upload_rgb_frame(tex_left, frame_left);
            upload_rgb_frame(tex_ref, frame_ref);
            upload_rgb_frame(tex_right, frame_right);
        }

        render_pyramid(&mut window, &shader, tex_left, tex_ref, tex_right);

        swap_timestamps.push(Instant::now());
        window.swap_buffers();
        step += 1;

        if use_pacer {
            let target_time = start_time + std::time::Duration::from_nanos(step as u64 * 4_166_667);
            while Instant::now() < target_time {
                std::hint::spin_loop();
            }
        }
    }

    let t_end_presentation = Instant::now();
    let total_presentation_dur = t_end_presentation.duration_since(start_time).as_secs_f64();
    let actual_fps = if total_presentation_dur > 0.0 && step > 0 {
        step as f64 / total_presentation_dur
    } else {
        240.0
    };

    unsafe {
        gl::DeleteTextures(1, &tex_left);
        gl::DeleteTextures(1, &tex_ref);
        gl::DeleteTextures(1, &tex_right);
    }
    drop(window);

    // Frame drop and jitter analysis
    let mut dropped_frames = 0;
    let mut max_interval_ms = 0.0f64;
    let mut min_interval_ms = f64::MAX;
    let mut total_intervals_ms = 0.0f64;
    let mut intervals = Vec::new();
    let mut stutters = Vec::new();

    if swap_timestamps.len() > 1 {
        for i in 1..swap_timestamps.len() {
            let diff = swap_timestamps[i].duration_since(swap_timestamps[i - 1]).as_secs_f64() * 1000.0;
            intervals.push(diff);
            if diff > max_interval_ms {
                max_interval_ms = diff;
            }
            if diff < min_interval_ms {
                min_interval_ms = diff;
            }
            total_intervals_ms += diff;
            if diff > 6.25 {
                dropped_frames += 1;
                stutters.push((i, diff));
            }
        }
    }

    let mean_interval_ms = if !intervals.is_empty() {
        total_intervals_ms / intervals.len() as f64
    } else {
        0.0
    };

    let variance = if intervals.len() > 1 {
        let mean = mean_interval_ms;
        let sum_sq_diff: f64 = intervals.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_sq_diff / (intervals.len() - 1) as f64
    } else {
        0.0
    };
    let jitter_ms = variance.sqrt();
    let dropped_pct = if !intervals.is_empty() {
        (dropped_frames as f64 / intervals.len() as f64) * 100.0
    } else {
        0.0
    };

    println!("\n=================================================================");
    println!("  LZ4 PLAYBACK BENCHMARK RESULTS");
    println!("=================================================================");
    println!("Total Frames Played : {} frames", step);
    println!("Average Playback FPS: {:.2} FPS", actual_fps);
    println!("Load & Decompress   : {:.4} seconds", t_load);
    println!("Decompress Speed    : {:.2} GB/s aggregate throughput", decompress_throughput_gb_s);
    println!("Frame Lock Efficiency: {:.2}%", (actual_fps / 240.0) * 100.0);
    println!("-----------------------------------------------------------------");
    println!("  FRAME TIMING & DROPPED FRAMES ANALYSIS");
    println!("-----------------------------------------------------------------");
    println!("Mean Frame Interval : {:.3} ms (Ideal: 4.167 ms)", mean_interval_ms);
    println!("Min Frame Interval  : {:.3} ms", if min_interval_ms == f64::MAX { 0.0 } else { min_interval_ms });
    println!("Max Frame Interval  : {:.3} ms (Worst Spike)", max_interval_ms);
    println!("Frame Timing Jitter : {:.3} ms (StdDev)", jitter_ms);
    println!("Total Dropped Frames: {} / {} ({:.2}%)", dropped_frames, intervals.len(), dropped_pct);

    if dropped_frames > 0 {
        println!("-----------------------------------------------------------------");
        println!("  DETAILED LIST OF DROPPED FRAMES (Worst 10)");
        println!("-----------------------------------------------------------------");
        let mut sorted_stutters = stutters.clone();
        sorted_stutters.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, (frame, interval)) in sorted_stutters.iter().take(10).enumerate() {
            println!("  #{:<2} | Frame Index: {:<4} | Delay: {:6.2} ms (Ideal: 4.17 ms)", idx + 1, frame, interval);
        }
    }
    println!("=================================================================\n");
}
