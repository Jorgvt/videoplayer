use std::env;
use std::ffi::CString;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use glfw::{Action, Context, Key, WindowHint};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const Y_SIZE: usize = WIDTH * HEIGHT;
const UV_WIDTH: usize = WIDTH / 2;
const UV_HEIGHT: usize = HEIGHT / 2;
const UV_SIZE: usize = UV_WIDTH * UV_HEIGHT;
const FRAME_SIZE: usize = Y_SIZE + UV_SIZE + UV_SIZE; // YUV420P frame size = 1,382,400 bytes

#[derive(Debug, Clone)]
struct VideoStreamData {
    num_frames: usize,
    data: Arc<Vec<u8>>,
}

fn get_dataset_dir() -> PathBuf {
    let default_path = PathBuf::from("/home/jv495/Datasets/GAIM240");
    if default_path.exists() {
        default_path
    } else {
        PathBuf::from("GAIM240")
    }
}

fn decode_video_cmd(path: &Path) -> Arc<VideoStreamData> {
    use std::process::Command;
    #[cfg(target_os = "windows")]
    let ffmpeg_bin = if std::path::Path::new("rust_player/lib/windows/bin/ffmpeg.exe").exists() {
        "rust_player/lib/windows/bin/ffmpeg.exe"
    } else if std::path::Path::new("lib/windows/bin/ffmpeg.exe").exists() {
        "lib/windows/bin/ffmpeg.exe"
    } else {
        "ffmpeg"
    };

    #[cfg(not(target_os = "windows"))]
    let ffmpeg_bin = if Path::new("rust_player/lib/usr/bin/ffmpeg").exists() {
        "rust_player/lib/usr/bin/ffmpeg"
    } else if Path::new("lib/usr/bin/ffmpeg").exists() {
        "lib/usr/bin/ffmpeg"
    } else {
        "ffmpeg"
    };

    let mut cmd = Command::new(ffmpeg_bin);

    #[cfg(target_os = "linux")]
    cmd.env(
        "LD_LIBRARY_PATH",
        "rust_player/lib/usr/lib/x86_64-linux-gnu:lib/usr/lib/x86_64-linux-gnu",
    );

    #[cfg(target_os = "windows")]
    {
        let dll_dir = "rust_player/lib/windows/bin";
        let current_path = std::env::var("PATH").unwrap_or_default();
        cmd.env("PATH", format!("{};{}", dll_dir, current_path));
    }

    let output = cmd
        .args([
            "-hwaccel",
            "cuda",
            "-c:v",
            "hevc_cuvid",
            "-i",
            &path.to_string_lossy(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "pipe:1",
        ])
        .output();

    let mut raw_data = Vec::new();
    if let Ok(out) = output {
        raw_data = out.stdout;
    }

    let num_frames = raw_data.len() / FRAME_SIZE;

    Arc::new(VideoStreamData {
        num_frames,
        data: Arc::new(raw_data),
    })
}

struct YuvQuadShader {
    program: u32,
    vbo: u32,
    u_y_loc: i32,
    u_u_loc: i32,
    u_v_loc: i32,
}

impl YuvQuadShader {
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
            uniform sampler2D u_tex_y;
            uniform sampler2D u_tex_u;
            uniform sampler2D u_tex_v;
            varying vec2 v_texcoord;
            void main() {
                float y = texture2D(u_tex_y, v_texcoord).r;
                float u = texture2D(u_tex_u, v_texcoord).r - 0.5;
                float v = texture2D(u_tex_v, v_texcoord).r - 0.5;
                
                float r = y + 1.402 * v;
                float g = y - 0.344136 * u - 0.714136 * v;
                float b = y + 1.772 * u;
                
                gl_FragColor = vec4(r, g, b, 1.0);
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

            let u_y_loc = gl::GetUniformLocation(program, CString::new("u_tex_y").unwrap().as_ptr());
            let u_u_loc = gl::GetUniformLocation(program, CString::new("u_tex_u").unwrap().as_ptr());
            let u_v_loc = gl::GetUniformLocation(program, CString::new("u_tex_v").unwrap().as_ptr());

            YuvQuadShader {
                program,
                vbo,
                u_y_loc,
                u_u_loc,
                u_v_loc,
            }
        }
    }

    fn draw_quad(&self, x1: f32, y1: f32, x2: f32, y2: f32, tex_y: u32, tex_u: u32, tex_v: u32) {
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
            gl::BindTexture(gl::TEXTURE_2D, tex_y);
            gl::Uniform1i(self.u_y_loc, 0);

            gl::ActiveTexture(gl::TEXTURE1);
            gl::BindTexture(gl::TEXTURE_2D, tex_u);
            gl::Uniform1i(self.u_u_loc, 1);

            gl::ActiveTexture(gl::TEXTURE2);
            gl::BindTexture(gl::TEXTURE_2D, tex_v);
            gl::Uniform1i(self.u_v_loc, 2);

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

struct YuvTextures {
    y: u32,
    u: u32,
    v: u32,
}

fn create_yuv_textures() -> YuvTextures {
    let mut textures = [0u32; 3];
    unsafe {
        gl::Enable(gl::TEXTURE_2D);
        gl::GenTextures(3, textures.as_mut_ptr());

        // Y Texture (1280x720)
        gl::BindTexture(gl::TEXTURE_2D, textures[0]);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexImage2D(
            gl::TEXTURE_2D, 0, gl::RED as i32, WIDTH as i32, HEIGHT as i32, 0,
            gl::RED, gl::UNSIGNED_BYTE, std::ptr::null()
        );

        // U Texture (640x360)
        gl::BindTexture(gl::TEXTURE_2D, textures[1]);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexImage2D(
            gl::TEXTURE_2D, 0, gl::RED as i32, UV_WIDTH as i32, UV_HEIGHT as i32, 0,
            gl::RED, gl::UNSIGNED_BYTE, std::ptr::null()
        );

        // V Texture (640x360)
        gl::BindTexture(gl::TEXTURE_2D, textures[2]);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexImage2D(
            gl::TEXTURE_2D, 0, gl::RED as i32, UV_WIDTH as i32, UV_HEIGHT as i32, 0,
            gl::RED, gl::UNSIGNED_BYTE, std::ptr::null()
        );
    }
    YuvTextures {
        y: textures[0],
        u: textures[1],
        v: textures[2],
    }
}

fn upload_yuv_frame(tex: &YuvTextures, raw_data: &[u8], frame_idx: usize) {
    let frame_offset = frame_idx * FRAME_SIZE;
    if frame_offset + FRAME_SIZE > raw_data.len() {
        return;
    }

    let y_ptr = &raw_data[frame_offset];
    let u_ptr = &raw_data[frame_offset + Y_SIZE];
    let v_ptr = &raw_data[frame_offset + Y_SIZE + UV_SIZE];

    unsafe {
        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D, tex.y);
        gl::TexSubImage2D(
            gl::TEXTURE_2D, 0, 0, 0, WIDTH as i32, HEIGHT as i32,
            gl::RED, gl::UNSIGNED_BYTE, y_ptr as *const _ as *const _,
        );

        gl::ActiveTexture(gl::TEXTURE1);
        gl::BindTexture(gl::TEXTURE_2D, tex.u);
        gl::TexSubImage2D(
            gl::TEXTURE_2D, 0, 0, 0, UV_WIDTH as i32, UV_HEIGHT as i32,
            gl::RED, gl::UNSIGNED_BYTE, u_ptr as *const _ as *const _,
        );

        gl::ActiveTexture(gl::TEXTURE2);
        gl::BindTexture(gl::TEXTURE_2D, tex.v);
        gl::TexSubImage2D(
            gl::TEXTURE_2D, 0, 0, 0, UV_WIDTH as i32, UV_HEIGHT as i32,
            gl::RED, gl::UNSIGNED_BYTE, v_ptr as *const _ as *const _,
        );
    }
}

fn render_pyramid_yuv(
    window: &mut glfw::Window,
    shader: &YuvQuadShader,
    tex_a: &YuvTextures,
    tex_ref: &YuvTextures,
    tex_c: &YuvTextures,
    data_a: &VideoStreamData,
    data_ref: &VideoStreamData,
    data_c: &VideoStreamData,
    frame_idx: usize,
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

    let idx_a = if data_a.num_frames > 0 { frame_idx % data_a.num_frames } else { 0 };
    let idx_ref = if data_ref.num_frames > 0 { frame_idx % data_ref.num_frames } else { 0 };
    let idx_c = if data_c.num_frames > 0 { frame_idx % data_c.num_frames } else { 0 };

    upload_yuv_frame(tex_a, &data_a.data, idx_a);
    upload_yuv_frame(tex_ref, &data_ref.data, idx_ref);
    upload_yuv_frame(tex_c, &data_c.data, idx_c);

    unsafe {
        gl::Viewport(0, 0, w, h);
        gl::ClearColor(0.02, 0.02, 0.03, 1.0);
        gl::Clear(gl::COLOR_BUFFER_BIT);

        gl::Viewport(x_offset, y_offset, w_view, h_view);

        shader.draw_quad(-0.5, 0.0, 0.5, 1.0, tex_ref.y, tex_ref.u, tex_ref.v);
        shader.draw_quad(-1.0, -1.0, 0.0, 0.0, tex_a.y, tex_a.u, tex_a.v);
        shader.draw_quad(0.0, -1.0, 1.0, 0.0, tex_c.y, tex_c.u, tex_c.v);
    }
}

struct FrameProfileRecord {
    frame_index: usize,
    get_image_ms: f64,
    upload_draw_ms: f64,
    flip_ms: f64,
    total_loop_ms: f64,
    swap_timestamp_sec: f64,
    inter_frame_interval_ms: Option<f64>,
}

fn calc_stats(vals: &[f64]) -> (f64, f64, f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    let std_dev = var.sqrt();
    (mean, std_dev, min, max)
}

fn find_scene_videos(
    dataset_dir: &Path,
    scene: &str,
    requested_metric: Option<&str>,
) -> Option<(PathBuf, PathBuf, PathBuf, String)> {
    let ref_path = dataset_dir.join(format!("{}_reference.mp4", scene));
    if !ref_path.exists() {
        println!("Error: Reference video not found at {:?}", ref_path);
        return None;
    }

    let prefix = format!("{}_", scene);
    let mut available_metrics = Vec::new();

    if let Ok(entries) = fs::read_dir(dataset_dir) {
        for entry in entries.flatten() {
            let filename = entry.file_name().to_string_lossy().to_string();
            if filename.starts_with(&prefix) && filename.ends_with(".mp4") {
                let rest = &filename[prefix.len()..filename.len() - 4];
                if rest != "reference" {
                    let metric = if let Some(idx) = rest.find("_level") {
                        &rest[..idx]
                    } else {
                        rest
                    };
                    if !available_metrics.contains(&metric.to_string()) {
                        available_metrics.push(metric.to_string());
                    }
                }
            }
        }
    }

    available_metrics.sort();

    if available_metrics.is_empty() {
        println!("Error: No distortion metrics found for scene '{}'", scene);
        return None;
    }

    let selected_metric = if let Some(req) = requested_metric {
        if available_metrics.iter().any(|m| m == req) {
            req.to_string()
        } else {
            println!("Warning: Requested metric '{}' not found. Available metrics: {:?}", req, available_metrics);
            available_metrics[0].clone()
        }
    } else if available_metrics.iter().any(|m| m == "temporal-resolution-multiplexing") {
        "temporal-resolution-multiplexing".to_string()
    } else {
        available_metrics[0].clone()
    };

    let level1_path = dataset_dir.join(format!("{}_{}_level1.mp4", scene, selected_metric));
    let level2_path = dataset_dir.join(format!("{}_{}_level2.mp4", scene, selected_metric));

    Some((ref_path, level1_path, level2_path, selected_metric))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut no_vsync = false;
    let mut use_pacer = false;
    let mut borderless = false;
    let mut windowed = false;
    let mut target_scene = "marbles".to_string();
    let mut requested_metric: Option<String> = None;
    let mut csv_out_path = "profile_results.csv".to_string();

    for arg in &args[1..] {
        if arg == "--no-vsync" || arg == "--novsync" || arg == "--uncapped" {
            no_vsync = true;
        } else if arg == "--pacer" || arg == "--pace-240" {
            no_vsync = true;
            use_pacer = true;
        } else if arg == "--borderless" {
            borderless = true;
        } else if arg == "--windowed" {
            windowed = true;
        } else if arg.starts_with("--scene=") {
            target_scene = arg.trim_start_matches("--scene=").to_string();
        } else if arg.starts_with("--metric=") {
            requested_metric = Some(arg.trim_start_matches("--metric=").to_string());
        } else if arg.starts_with("--out=") {
            csv_out_path = arg.trim_start_matches("--out=").to_string();
        }
    }

    let dataset_dir = get_dataset_dir();
    println!("=== GAIM240 Rust Video Player Multi-Stage Profiler ===");
    println!("Dataset directory: {:?}", dataset_dir);
    println!("Target scene     : {}", target_scene);
    println!(
        "VSync / Pacer    : {}",
        if use_pacer {
            "SOFTWARE PACER (--pacer 240.000 FPS Locked)"
        } else if no_vsync {
            "DISABLED (--no-vsync Raw FPS Max Throughput)"
        } else {
            "ENABLED (Locked to Hardware Refresh)"
        }
    );
    println!("Output CSV Path  : {}", csv_out_path);

    let (ref_path, level1_path, level2_path, selected_metric) = match find_scene_videos(&dataset_dir, &target_scene, requested_metric.as_deref()) {
        Some(res) => res,
        None => return,
    };

    println!("Selected metric  : {}", selected_metric);
    println!("  Left Video (A) : {:?}", level1_path.file_name().unwrap_or_default());
    println!("  Center (Ref)   : {:?}", ref_path.file_name().unwrap_or_default());
    println!("  Right Video (C): {:?}", level2_path.file_name().unwrap_or_default());

    println!("\n[1/2] Preloading video frames into RAM (YUV420P Optimized Pipeline)...");
    let t_preload_start = Instant::now();

    let p_a = level1_path.clone();
    let p_ref = ref_path.clone();
    let p_c = level2_path.clone();

    let h_a = thread::spawn(move || decode_video_cmd(&p_a));
    let h_ref = thread::spawn(move || decode_video_cmd(&p_ref));
    let h_c = thread::spawn(move || decode_video_cmd(&p_c));

    let stream_a = h_a.join().unwrap();
    let stream_ref = h_ref.join().unwrap();
    let stream_c = h_c.join().unwrap();

    let t_preload_elapsed = t_preload_start.elapsed().as_secs_f64();
    println!(
        "Preload complete in {:.3} s! Stream A: {} frames, Ref: {} frames, Stream C: {} frames",
        t_preload_elapsed, stream_a.num_frames, stream_ref.num_frames, stream_c.num_frames
    );

    if stream_a.num_frames == 0 || stream_ref.num_frames == 0 || stream_c.num_frames == 0 {
        println!("Error: Failed to decode frames for one or more video streams.");
        return;
    }

    let total_frames = stream_a.num_frames.min(stream_ref.num_frames).min(stream_c.num_frames);
    if total_frames == 0 {
        println!("Error: No valid frame sequence decoded.");
        return;
    }

    println!("\n[2/2] Initializing OpenGL Context & Starting Profiling Run...");
    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    glfw.window_hint(WindowHint::Resizable(true));
    glfw.window_hint(WindowHint::DoubleBuffer(true));
    glfw.window_hint(WindowHint::ContextVersion(2, 1));

    let (mon_w, mon_h) = glfw.with_primary_monitor(|_, m| {
        m.and_then(|mon| mon.get_video_mode())
            .map(|mode| (mode.width as u32, mode.height as u32))
            .unwrap_or((1280, 720))
    });

    let (mut window, events) = if borderless {
        glfw.window_hint(WindowHint::Decorated(false));
        let (mut w, ev) = glfw
            .create_window(
                mon_w,
                mon_h,
                "GAIM240 Rust Triple Player Profiler",
                glfw::WindowMode::Windowed,
            )
            .expect("Failed to create borderless window");
        w.set_pos(0, 0);
        (w, ev)
    } else if windowed {
        glfw.create_window(
            1280,
            720,
            "GAIM240 Rust Triple Player Profiler",
            glfw::WindowMode::Windowed,
        )
        .expect("Failed to create windowed GLFW window")
    } else {
        glfw.with_connected_monitors(|glfw_ref, monitors| {
            if let Some(m) = monitors.first() {
                glfw_ref
                    .create_window(
                        mon_w,
                        mon_h,
                        "GAIM240 Rust Triple Player Profiler",
                        glfw::WindowMode::FullScreen(m),
                    )
                    .expect("Failed to create fullscreen window")
            } else {
                glfw_ref
                    .create_window(
                        1280,
                        720,
                        "GAIM240 Rust Triple Player Profiler",
                        glfw::WindowMode::Windowed,
                    )
                    .expect("Failed to create windowed window")
            }
        })
    };

    window.make_current();
    if no_vsync {
        glfw.set_swap_interval(glfw::SwapInterval::None);
    } else {
        glfw.set_swap_interval(glfw::SwapInterval::Sync(1));
    }
    window.set_key_polling(true);

    gl::load_with(|s| window.get_proc_address(s) as *const _);

    let shader = YuvQuadShader::new();
    let tex_a = create_yuv_textures();
    let tex_ref = create_yuv_textures();
    let tex_c = create_yuv_textures();

    let mut records: Vec<FrameProfileRecord> = Vec::with_capacity(total_frames);
    let start_instant = Instant::now();

    println!("\n=== Rust Triple OpenGL Player Profiling Active ===");
    println!("Profiling {} frames...", total_frames);
    println!("Press [Q] or [Esc] to interrupt early.\n");

    let mut step_count = 0;
    let mut prev_swap_sec: Option<f64> = None;

    while !window.should_close() && step_count < total_frames {
        let t0 = Instant::now();

        // 1. Image retrieval from contiguous RAM block
        let t1 = Instant::now();

        // 2. Upload YUV textures & render on GPU
        render_pyramid_yuv(
            &mut window,
            &shader,
            &tex_a,
            &tex_ref,
            &tex_c,
            &stream_a,
            &stream_ref,
            &stream_c,
            step_count,
        );

        let t2 = Instant::now();

        // 3. Swap buffers (VSync wait occurs here if enabled)
        window.swap_buffers();

        let t3 = Instant::now();

        let get_image_ms = (t1 - t0).as_secs_f64() * 1000.0;
        let upload_draw_ms = (t2 - t1).as_secs_f64() * 1000.0;
        let flip_ms = (t3 - t2).as_secs_f64() * 1000.0;
        let total_loop_ms = (t3 - t0).as_secs_f64() * 1000.0;
        let swap_timestamp_sec = (t3 - start_instant).as_secs_f64();

        let inter_frame_interval_ms = prev_swap_sec.map(|prev| (swap_timestamp_sec - prev) * 1000.0);
        prev_swap_sec = Some(swap_timestamp_sec);

        records.push(FrameProfileRecord {
            frame_index: step_count,
            get_image_ms,
            upload_draw_ms,
            flip_ms,
            total_loop_ms,
            swap_timestamp_sec,
            inter_frame_interval_ms,
        });

        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            if let glfw::WindowEvent::Key(Key::Escape | Key::Q, _, Action::Press, _) = event {
                println!("\nProfiling interrupted by user keypress.");
                window.set_should_close(true);
            }
        }

        if use_pacer {
            let target_time = start_instant + std::time::Duration::from_nanos((step_count as u64 + 1) * 4_166_667);
            while Instant::now() < target_time {
                std::hint::spin_loop();
            }
        }

        step_count += 1;
    }

    let elapsed_total = start_instant.elapsed().as_secs_f64();
    let num_records = records.len();

    println!("\n=========================================================");
    println!("              RUST PROFILER SUMMARY RESULTS              ");
    println!("=========================================================");
    println!("Total Frames Profiled: {} frames", num_records);
    println!("Total Elapsed Time   : {:.4} s", elapsed_total);
    if elapsed_total > 0.0 {
        println!("Average Presentation FPS: {:.2} FPS", num_records as f64 / elapsed_total);
    }

    let get_img_vec: Vec<f64> = records.iter().map(|r| r.get_image_ms).collect();
    let up_draw_vec: Vec<f64> = records.iter().map(|r| r.upload_draw_ms).collect();
    let flip_vec: Vec<f64> = records.iter().map(|r| r.flip_ms).collect();
    let total_vec: Vec<f64> = records.iter().map(|r| r.total_loop_ms).collect();
    let inter_vec: Vec<f64> = records.iter().filter_map(|r| r.inter_frame_interval_ms).collect();

    let (s_get_m, s_get_s, s_get_min, s_get_max) = calc_stats(&get_img_vec);
    let (s_up_m, s_up_s, s_up_min, s_up_max) = calc_stats(&up_draw_vec);
    let (s_fl_m, s_fl_s, s_fl_min, s_fl_max) = calc_stats(&flip_vec);
    let (s_tot_m, s_tot_s, s_tot_min, s_tot_max) = calc_stats(&total_vec);
    let (s_int_m, s_int_s, s_int_min, s_int_max) = calc_stats(&inter_vec);

    println!("\nStage Breakdown (mean ± std | min .. max):");
    println!("  1. Fetch Frame (RAM) : {:6.3} ± {:5.3} ms  [{:6.3} .. {:6.3} ms]", s_get_m, s_get_s, s_get_min, s_get_max);
    println!("  2. Upload & Draw GPU: {:6.3} ± {:5.3} ms  [{:6.3} .. {:6.3} ms]", s_up_m, s_up_s, s_up_min, s_up_max);
    println!("  3. VSync Flip Wait  : {:6.3} ± {:5.3} ms  [{:6.3} .. {:6.3} ms]", s_fl_m, s_fl_s, s_fl_min, s_fl_max);
    println!("  ---------------------------------------------------------");
    println!("  Total Loop Cycle   : {:6.3} ± {:5.3} ms  [{:6.3} .. {:6.3} ms]", s_tot_m, s_tot_s, s_tot_min, s_tot_max);
    println!("  Inter-Frame Interval: {:6.3} ± {:5.3} ms  [{:6.3} .. {:6.3} ms]", s_int_m, s_int_s, s_int_min, s_int_max);

    // Save CSV matching Python profiler schema
    let mut file = File::create(&csv_out_path).expect("Failed to create CSV output file");
    writeln!(file, "FrameIndex,GetImageDuration_ms,UploadDrawDuration_ms,FlipDuration_ms,TotalLoopDuration_ms,SwapTimestamp_sec,InterFrameInterval_ms").unwrap();

    for record in &records {
        let inter_str = match record.inter_frame_interval_ms {
            Some(val) => format!("{:.6}", val),
            None => "N/A".to_string(),
        };
        writeln!(
            file,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            record.frame_index,
            record.get_image_ms,
            record.upload_draw_ms,
            record.flip_ms,
            record.total_loop_ms,
            record.swap_timestamp_sec,
            inter_str
        )
        .unwrap();
    }

    println!("\nProfiling CSV saved to: {}", csv_out_path);
    println!("You can now plot this data with Python using: uv run plot_profile.py {}", csv_out_path);
    println!("=========================================================");
}
