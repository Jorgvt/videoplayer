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

fn get_png_dataset_dir() -> PathBuf {
    if let Ok(env_val) = std::env::var("GAIM240_PNG_DIR") {
        let p = PathBuf::from(env_val);
        if p.exists() {
            return p;
        }
    }
    for fallback in &["../GAIM240_refs_png", "D:\\GAIM240_refs_png", "C:\\GAIM240_refs_png"] {
        let p = PathBuf::from(fallback);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("GAIM240_refs_png")
}

fn get_png_files(dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() && p.extension().map_or(false, |ext| ext == "png") {
                paths.push(p);
            }
        }
    }
    paths.sort();
    paths
}

fn decode_png_to_rgb(path: &Path) -> Result<Vec<u8>, String> {
    let file = fs::File::open(path).map_err(|e| format!("Failed to open file {:?}: {}", path, e))?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().map_err(|e| format!("Failed to read PNG info for {:?}: {}", path, e))?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("Failed to read PNG frame for {:?}: {}", path, e))?;
    let bytes = &buf[..info.buffer_size()];

    let rgb_data = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity(bytes.len() / 4 * 3);
            for chunk in bytes.chunks_exact(4) {
                rgb.extend_from_slice(&chunk[0..3]);
            }
            rgb
        }
        _ => return Err(format!("Unsupported color type {:?} for {:?}", info.color_type, path)),
    };

    if rgb_data.len() != FRAME_SIZE {
        return Err(format!("Unexpected frame size {} for {:?} (expected {})", rgb_data.len(), path, FRAME_SIZE));
    }

    Ok(rgb_data)
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

            gl::DeleteShader(vert_shader);
            gl::DeleteShader(frag_shader);

            let mut vbo = 0;
            gl::GenBuffers(1, &mut vbo);

            let u_tex_loc = gl::GetUniformLocation(program, CString::new("u_tex").unwrap().as_ptr());

            RgbQuadShader {
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
                0 as *const _,
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
    let (w, h) = window.get_framebuffer_size();
    let target_aspect = 16.0 / 9.0;

    let mut w_view = w;
    let mut h_view = (w as f32 / target_aspect) as i32;
    if h_view > h {
        h_view = h;
        w_view = (h as f32 * target_aspect) as i32;
    }
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

    let dataset_dir = get_png_dataset_dir();
    if !dataset_dir.exists() {
        println!("Error: Lossless PNG dataset directory not found at {:?}", dataset_dir);
        return;
    }

    // Discover the first 3 directories to serve as Left, Ref, and Right reference videos
    let mut scenes = Vec::new();
    if let Ok(entries) = fs::read_dir(&dataset_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                scenes.push(p);
            }
        }
    }
    scenes.sort();

    if scenes.len() < 3 {
        println!("Error: Need at least 3 scene directories inside {:?} for benchmark, found {}", dataset_dir, scenes.len());
        return;
    }

    let left_scene_dir = &scenes[0];
    let ref_scene_dir = &scenes[1];
    let right_scene_dir = &scenes[2];

    println!("\n=================================================================");
    # [rustfmt::skip]
    println!("  GAIM240 LOSSLESS PNG SEQUENCE BENCHMARK PLAYER");
    println!("=================================================================");
    println!("Dataset Dir     : {:?}", dataset_dir);
    println!("Left Stream Src : {:?}", left_scene_dir.file_name().unwrap());
    println!("Ref Stream Src  : {:?}", ref_scene_dir.file_name().unwrap());
    println!("Right Stream Src: {:?}", right_scene_dir.file_name().unwrap());
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

    // 1. Load and decompress PNG files in parallel
    println!("Step 1: Discovering PNG files...");
    let left_paths = get_png_files(left_scene_dir);
    let ref_paths = get_png_files(ref_scene_dir);
    let right_paths = get_png_files(right_scene_dir);

    let num_frames = left_paths.len().min(ref_paths.len()).min(right_paths.len()).min(1200);
    if num_frames == 0 {
        println!("Error: Found 0 PNG frames in one of the scene directories.");
        return;
    }
    println!("Found {} frames to load per stream.", num_frames);

    println!("Step 2: Pre-decoding PNG sequences to system RAM in parallel...");
    let t_load_start = Instant::now();

    let decode_stream = |paths: &[PathBuf]| -> Vec<Vec<u8>> {
        paths[..num_frames]
            .par_iter()
            .map(|path| {
                decode_png_to_rgb(path).unwrap_or_else(|e| {
                    panic!("Error decoding PNG: {}", e);
                })
            })
            .collect()
    };

    // Parallel decoding of all 3 streams concurrently using Rayon threadpool
    let (left_frames, (ref_frames, right_frames)) = rayon::join(
        || decode_stream(&left_paths),
        || rayon::join(|| decode_stream(&ref_paths), || decode_stream(&right_paths)),
    );

    let t_load = t_load_start.elapsed().as_secs_f64();
    let total_ram_mb = (num_frames * 3 * FRAME_SIZE) as f64 / 1024.0 / 1024.0;
    println!(
        "Pre-decode Complete: loaded {} frames in {:.2}s ({:.2} MB raw pixels in RAM)",
        num_frames * 3,
        t_load,
        total_ram_mb
    );

    // 2. Initialize GLFW and OpenGL
    println!("Step 3: Initializing graphics context...");
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
                .create_window(mon_w, mon_h, "GAIM240 Lossless PNG Visualizer", glfw::WindowMode::FullScreen(mon))
                .unwrap()
        } else {
            glfw_ref
                .create_window(1280, 720, "GAIM240 Lossless PNG Visualizer", glfw::WindowMode::Windowed)
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
    println!("Step 4: Starting playback loop...");
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

        let frame_left = &left_frames[step];
        let frame_ref = &ref_frames[step];
        let frame_right = &right_frames[step];

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

    let total_presentation_dur = start_time.elapsed().as_secs_f64();
    let mut actual_fps = 239.76;
    if total_presentation_dur > 0.0 && step > 0 {
        actual_fps = step as f64 / total_presentation_dur;
    }

    // Clean up textures
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
    let mut stutters = Vec::new(); // stores (frame_index, interval_ms)

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
            
            // Frame drop threshold: ideal is 4.167 ms. Threshold is 1.5 * 4.167 = 6.25 ms.
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
    println!("  PNG PLAYBACK BENCHMARK RESULTS");
    println!("=================================================================");
    println!("Total Frames Played : {} frames", step);
    println!("Average Playback FPS: {:.2} FPS", actual_fps);
    println!("Load & Decode Time  : {:.4} seconds", t_load);
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
            println!("  #{:<2} | Frame Index: {:<4} | Delay: {:>6.2} ms (Ideal: 4.17 ms)", idx + 1, frame, interval);
        }
    }
    println!("=================================================================\n");
}
